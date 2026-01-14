from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages as django_messages
from .models import UploadedFile, Message, UserPreference
from django.http import HttpResponseForbidden, JsonResponse, StreamingHttpResponse
import os
import logging
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from django.views.decorators.csrf import csrf_exempt
import json
from django import forms
import requests
from django.utils import timezone
import re
from django.conf import settings
import threading
from django.core.cache import cache

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)  # Add override=True to ensure it reloads

# Initialize Groq API - get key from environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY environment variable is required")
logger.info("Using Groq API key from environment variables")

# Dictionary to track which models are available/working
AVAILABLE_MODELS = {
    "llama-3.1-8b-instant": True,  # Default model is always assumed to be available
    "llama3-8b-8192": True,  # Assume available until proven otherwise
    "llama-3.3-70b-versatile": True,  # This is a known good model
}

# Model-specific configurations
MODEL_CONFIGS = {
    "llama-3.1-8b-instant": {
        "max_tokens": 8192,
        "context_window": 128000,
        "timeout": 30,
        "rate_limit_tokens": 50000  # Higher rate limits
    },
    "llama3-8b-8192": {
        "max_tokens": 8192,
        "context_window": 8192,
        "timeout": 30,
        "rate_limit_tokens": 6000  # Lower rate limits
    },
    "llama-3.3-70b-versatile": {
        "max_tokens": 32767,
        "context_window": 128000,
        "timeout": 60,
        "rate_limit_tokens": 10000  # Medium rate limits
    }
}

def get_model_config(model_name):
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name, {
        "max_tokens": 32767,  # Default values
        "context_window": 32767,
        "timeout": 60
    })

def check_model_availability(model_name, timeout=30):
    """Check if a model is available using a simple API call"""
    global AVAILABLE_MODELS
    
    # Always consider models available for now to prevent fallback behavior
    # This helps users test their preferred model directly
    return True
    
    # The rest of this function is now bypassed
    # If we've already checked and know the status, return it
    if AVAILABLE_MODELS[model_name] is not None:
        return AVAILABLE_MODELS[model_name]
    
    # Check model availability with LangChain
    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=10,
            timeout=timeout,
            max_retries=3
        )
        
        # Simple test message
        message = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="hello")
        ]
        
        # Make request
        response = llm.invoke(message)
        
        # If we get here, the model is available
        AVAILABLE_MODELS[model_name] = True
        logger.info(f"Model {model_name} is available")
        return True
    except Exception as e:
        # If there's an error, mark the model as unavailable
        AVAILABLE_MODELS[model_name] = False
        logger.error(f"Model {model_name} is unavailable: {str(e)}")
        return False

# Model form for user preferences
class UserPreferenceForm(forms.ModelForm):
    class Meta:
        model = UserPreference
        fields = ['preferred_model', 'use_custom_prompt', 'custom_system_prompt', 'use_resources']
        widgets = {
            'preferred_model': forms.Select(attrs={'class': 'form-select'}),
            'use_custom_prompt': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'custom_system_prompt': forms.Textarea(attrs={'class': 'form-control', 'rows': 5, 'placeholder': 'You are a mental health counselor chatbot. You provide compassionate, evidence-based advice...'}),
            'use_resources': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
        labels = {
            'preferred_model': 'Preferred AI Model',
            'use_custom_prompt': 'Use Custom System Prompt',
            'custom_system_prompt': 'Custom System Prompt',
            'use_resources': 'Use Uploaded Resources',
        }
        help_texts = {
            'custom_system_prompt': 'Define how the AI should behave. This overrides the default system prompts.',
            'use_resources': 'When enabled, the AI will reference your uploaded documents when generating responses.',
        }

def get_user_llm(user):
    """Get the appropriate LLM instance based on user preferences"""
    try:
        # Get or create user preferences
        preference, created = UserPreference.objects.get_or_create(user=user)
        model_name = preference.preferred_model
        
        # Log additional details about the model
        logger.info(f"User {user.username} is using model: {model_name}")
        logger.info(f"Model choice details: {dict(UserPreference.MODEL_CHOICES).get(model_name, 'Unknown model')}")
        
        # For deepseek and qwen models, we'll use direct Groq API calls instead of langchain
        # This is handled in the send_message function
        
        # Log if this is a newly created preference
        if created:
            logger.info(f"Created new user preference for {user.username} with default model {model_name}")
        
        # Create and return the LLM instance with user's parameters
        return ChatGroq(
            api_key=groq_api_key,
            model=model_name,
            temperature=preference.temperature,
            max_tokens=preference.max_tokens,
            timeout=20,
            max_retries=3,
            streaming=preference.stream
        )
    except Exception as e:
        logger.error(f"Error creating LLM instance: {str(e)}")
        # Default to llama if there's an error
        return ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            max_tokens=4096,
            timeout=20,
            max_retries=3,
            streaming=True
        )

def get_chat_history(user, max_messages=10):
    # Convert database messages to format needed for Groq
    # Limit history to prevent token limit issues
    messages = []
    # Get the most recent messages, limiting to max_messages
    for msg in Message.objects.filter(user=user).order_by('-timestamp')[:max_messages][::-1]:
        if msg.is_bot:
            messages.append(AIMessage(content=msg.content))
        else:
            messages.append(HumanMessage(content=msg.content))
    return messages

@login_required
def chat_home(request):
    # Get user's chat history
    chat_messages = Message.objects.filter(user=request.user).order_by('timestamp')
    
    logger.info(f"Loading chat for user {request.user.username}. Found {chat_messages.count()} messages.")
    
    if not chat_messages.exists():
        # Create initial bot message for new users
        logger.info(f"Creating initial welcome message for new user {request.user.username}")
        Message.objects.create(
            user=request.user,
            content="Hi, I'm your mental health counselor. How can I help you today?",
            is_bot=True
        )
        chat_messages = Message.objects.filter(user=request.user)
    
    context = {
        'chat_messages': chat_messages,
        'user': request.user
    }
    return render(request, 'chat/chat_home.html', context)

@login_required
@csrf_exempt
def get_suggestions(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            current_input = data.get('current_input', '').strip()
            
            # Get user preferences
            preference, created = UserPreference.objects.get_or_create(user=request.user)
            
            # Use a low temperature for consistent results
            temperature = 0.2
            
            # Check if the suggestions should be regenerated
            force_refresh = data.get('force_refresh', False)
            
            # Create a user-specific cache key to avoid cross-user cache issues
            # Include whether custom prompt is used in the key to avoid switching
            use_custom = preference.use_custom_prompt and preference.custom_system_prompt
            user_id = request.user.id
            
            if use_custom:
                # Use a hash of the custom prompt for the cache key
                prompt_hash = hash(preference.custom_system_prompt)
                cache_key = f"suggestions_cache_user_{user_id}_custom_{prompt_hash}"
                persona = preference.custom_system_prompt
            else:
                # For default prompt, still have user-specific cache
                cache_key = f"suggestions_cache_user_{user_id}_default"
                persona = """You are a compassionate mental health counselor who provides supportive, evidence-based guidance. 
                You help users explore their thoughts and feelings, offer coping strategies, and provide information about mental health topics."""
            
            # Try to get cached suggestions
            if not force_refresh:
                cached_suggestions = cache.get(cache_key)
                if cached_suggestions:
                    logger.info(f"Using cached suggestions for user {request.user.username} with key {cache_key}")
                    return JsonResponse({
                        'status': 'success',
                        'suggestions': cached_suggestions
                    })
            
            # Create a very clear and direct system prompt for suggestion generation
            system_prompt = f"""PERSONA DEFINITION:
{persona}

TASK INSTRUCTIONS:
1. You are generating chat suggestions that perfectly match the PERSONA defined above.
2. These suggestions should be questions or prompts a user might want to ask the assistant.
3. The suggestions MUST deeply reflect the character, expertise, and focus areas in the PERSONA.
4. Consider the conversation context when generating relevant suggestions.
5. Generate 5 suggestions that would make sense for users to ask an assistant with this PERSONA.
6. Make sure suggestions are diverse and cover different topics relevant to the PERSONA.

FORMAT INSTRUCTIONS:
Return ONLY a JSON array of strings with no explanation, like:
["Suggestion 1", "Suggestion 2", "Suggestion 3", "Suggestion 4", "Suggestion 5"]

IMPORTANT: Make these suggestions clearly and strongly reflect the PERSONA's unique traits and specialization.
"""
            
            try:
                # Initialize the model client directly for more control
                client = Groq(api_key=groq_api_key)
                
                # Format the messages for the API request
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate suggestions that users would want to ask based on the PERSONA:"}
                ]
                
                # Make the API call with a low temperature for consistent results
                response = client.chat.completions.create(
                    model=preference.preferred_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=300
                )
                
                suggestions_text = response.choices[0].message.content.strip()
                
                # Try parsing the JSON array
                try:
                    # Try to parse as JSON
                    suggestions = json.loads(suggestions_text)
                    
                    # Validate that it's an array of strings
                    if isinstance(suggestions, list) and all(isinstance(item, str) for item in suggestions):
                        logger.info(f"Successfully generated {len(suggestions)} suggestions")
                        
                        # Cache the suggestions for 24 hours (longer cache to reduce API calls)
                        cache.set(cache_key, suggestions, 60 * 60 * 24)
                        
                        return JsonResponse({
                            'status': 'success',
                            'suggestions': suggestions
                        })
                    else:
                        # If not properly formatted, try to extract array-like content
                        cleaned_text = suggestions_text.strip()
                        if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
                            try:
                                # Try again with the cleaned text
                                suggestions = json.loads(cleaned_text)
                                if isinstance(suggestions, list):
                                    # Cache for 24 hours
                                    cache.set(cache_key, suggestions, 60 * 60 * 24)
                                    
                                    return JsonResponse({
                                        'status': 'success',
                                        'suggestions': suggestions
                                    })
                            except:
                                pass
                        
                        # Fall back to default suggestions
                        default_suggestions = [
                            "How can I manage my anxiety?",
                            "I've been feeling down lately",
                            "Help me with stress management",
                            "I need advice about my relationship",
                            "How do I deal with intrusive thoughts?"
                        ]
                        
                        # Cache default suggestions
                        cache.set(cache_key, default_suggestions, 60 * 60 * 24)
                            
                        return JsonResponse({
                            'status': 'success',
                            'suggestions': default_suggestions
                        })
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract an array-like structure
                    import re
                    array_match = re.search(r'\[(.*)\]', suggestions_text, re.DOTALL)
                    if array_match:
                        try:
                            array_content = array_match.group(0)
                            suggestions = json.loads(array_content)
                            if isinstance(suggestions, list) and len(suggestions) > 0:
                                # Cache for 24 hours
                                cache.set(cache_key, suggestions, 60 * 60 * 24)
                                    
                                return JsonResponse({
                                    'status': 'success',
                                    'suggestions': suggestions
                                })
                        except:
                            pass
                    
                    # If still no success, provide default suggestions
                    default_suggestions = [
                        "How can I manage my anxiety?",
                        "I've been feeling down lately",
                        "Help me with stress management",
                        "I need advice about my relationship",
                        "How do I deal with intrusive thoughts?"
                    ]
                    
                    # Cache default suggestions
                    cache.set(cache_key, default_suggestions, 60 * 60 * 24)
                        
                    return JsonResponse({
                        'status': 'success',
                        'suggestions': default_suggestions
                    })
                    
            except Exception as e:
                logger.error(f"Error getting suggestions from model: {str(e)}")
                # Return default suggestions on error
                default_suggestions = [
                    "How can I manage my anxiety?",
                    "I've been feeling down lately",
                    "Help me with stress management",
                    "I need advice about my relationship",
                    "How do I deal with intrusive thoughts?"
                ]
                
                return JsonResponse({
                    'status': 'success',
                    'suggestions': default_suggestions
                })
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@login_required
@csrf_exempt
def send_message(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            
            if not user_message:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Message cannot be empty'
                })
            
            # Save user message
            user_msg = Message.objects.create(
                user=request.user,
                content=user_message,
                is_bot=False
            )
            
            # Get user's preferred model and parameters
            try:
                preference, created = UserPreference.objects.get_or_create(user=request.user)
                current_model = "llama-3.1-8b-instant"  # Force use of this model
                
                # Get model configuration
                model_config = get_model_config(current_model)
                
                # Skip model availability check - always use the user's preferred model
                # This allows users to test any model they select
                
                logger.info(f"Using model {current_model} with config: {model_config}")
            except Exception as e:
                current_model = "llama-3.1-8b-instant"
                model_config = get_model_config(current_model)
                logger.error(f"Error getting user preference: {str(e)}")
            
            # Get chat history and prepare for Groq - limit based on model
            max_history = 5 if '8b' in current_model.lower() else 3
            chat_history = get_chat_history(request.user, max_history)
            
            # Create system prompts that are more distinctive to each model
            model_type = current_model.split('-')[0].lower()
            
            # If user has a custom system prompt and has enabled it, use that instead
            if preference.use_custom_prompt and preference.custom_system_prompt:
                system_prompt = preference.custom_system_prompt
                logger.info(f"Using custom system prompt for user {request.user.username}")
            else:
                # Use default prompts based on model
                if model_type == 'llama':
                    if '3.3' in current_model:
                        system_prompt = "You are a helpful mental health counselor chatbot powered by Llama 3.3. You provide compassionate, evidence-based advice with a focus on emotional intelligence."
                    elif '3.1' in current_model:
                        system_prompt = "You are a mental health counselor chatbot powered by Llama 3.1. You are fast, efficient, and provide clear, concise mental health guidance."
                    else:
                        system_prompt = "You are a mental health counselor chatbot powered by Llama 3. You specialize in clear, concise advice with a warm and supportive tone."
                else:
                    system_prompt = "You are a helpful mental health counselor chatbot."
            
            # Get relevant resources if enabled
            resource_content = ""
            if preference.use_resources:
                # Get processed content from user's uploaded files
                user_files = UploadedFile.objects.filter(user=request.user, processed=True).exclude(content__isnull=True).exclude(content="")
                
                if user_files.exists():
                    # First, try to find contextually relevant resources based on the user message
                    relevant_content = []
                    
                    # Use simple keyword matching to find relevant files
                    keywords = extract_keywords(user_message)
                    
                    for file in user_files:
                        # Calculate relevance score based on keyword matches
                        score = 0
                        for keyword in keywords:
                            if keyword.lower() in file.content.lower():
                                score += 1
                        
                        if score > 0:
                            # Extract relevant snippets around keywords
                            snippets = []
                            lines = file.content.split('\n')
                            
                            for keyword in keywords:
                                for i, line in enumerate(lines):
                                    if keyword.lower() in line.lower():
                                        # Get context around the matching line
                                        start = max(0, i - 2)
                                        end = min(len(lines), i + 3)
                                        context = "\n".join(lines[start:end])
                                        snippets.append(context)
                            
                            if snippets:
                                relevant_content.append({
                                    'file_name': file.name,
                                    'snippets': snippets[:3],  # Limit to 3 snippets per file
                                    'score': score
                                })
                    
                    # Sort by relevance score
                    relevant_content.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Build the resource content to include in the system prompt
                    if relevant_content:
                        resource_content = "REFERENCE INFORMATION FROM USER'S UPLOADED DOCUMENTS:\n\n"
                        for i, content in enumerate(relevant_content[:3]):  # Only use top 3 most relevant files
                            resource_content += f"DOCUMENT {i+1}: {content['file_name']}\n"
                            for j, snippet in enumerate(content['snippets']):
                                resource_content += f"Snippet {j+1}:\n{snippet}\n\n"
                    
                    # If no relevant content found, use summaries from the most recent files
                    if not resource_content and user_files.count() > 0:
                        resource_content = "GENERAL INFORMATION FROM USER'S UPLOADED DOCUMENTS:\n\n"
                        for i, file in enumerate(user_files.order_by('-uploaded_at')[:2]):
                            # Use first 1000 characters as a summary
                            summary = file.content[:1000] + "..." if len(file.content) > 1000 else file.content
                            resource_content += f"DOCUMENT {i+1}: {file.name}\n{summary}\n\n"
            
            # If we have resource content, add it to the system prompt
            if resource_content:
                system_prompt = f"{system_prompt}\n\n{resource_content}\n\nIMPORTANT: Use the above document information when relevant to your response, but don't explicitly mention you're using 'uploaded documents'. Just naturally incorporate the information as part of your knowledge."
            
            logger.info(f"Using system prompt: {system_prompt[:100]}...")
            
            try:
                # Initialize Groq client for streaming
                client = Groq(api_key=groq_api_key)
                
                # Prepare messages for the API call
                messages = [{"role": "system", "content": system_prompt}]
                
                # Add chat history
                for msg in chat_history:
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
                
                # Add current message
                messages.append({"role": "user", "content": user_message})
                
                # Log the request details
                logger.info(f"Making API request with model: {current_model}")
                logger.info(f"System prompt: {system_prompt}")
                logger.info(f"User message: {user_message}")
                logger.info(f"Temperature: {preference.temperature}")
                logger.info(f"Max tokens: {preference.max_tokens}")
                
                # Make API call with user's parameters
                if preference.stream:
                    # Streaming response
                    try:
                        stream = client.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            temperature=preference.temperature,
                            max_tokens=min(preference.max_tokens, model_config["max_tokens"]),
                            stream=True
                        )
                        
                        # Return StreamingHttpResponse
                        def generate():
                            full_response = ""
                            for chunk in stream:
                                if chunk.choices[0].delta.content is not None:
                                    content = chunk.choices[0].delta.content
                                    full_response += content
                                    yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"
                            
                            # Clean and format the response
                            formatted_response = format_bot_response(full_response)
                            
                            # Save the complete message to database
                            Message.objects.create(
                                user=request.user,
                                content=formatted_response,
                                is_bot=True
                            )
                            yield f"data: {json.dumps({'content': '', 'message': formatted_response, 'done': True})}\n\n"
                        
                        response = StreamingHttpResponse(
                            generate(),
                            content_type='text/event-stream'
                        )
                        response['X-Accel-Buffering'] = 'no'
                        return response
                    except Exception as e:
                        # Log detailed streaming error
                        logger.error(f"Streaming error with model {current_model}: {str(e)}")
                        logger.error(f"Error type: {type(e).__name__}")
                        logger.error(f"Error details: {str(e)}")
                        # Continue to non-streaming approach
                        preference.stream = False
                
                # Non-streaming response
                try:
                    response = client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        temperature=preference.temperature,
                        max_tokens=min(preference.max_tokens, model_config["max_tokens"]),
                        stream=False
                    )
                    
                    bot_message = response.choices[0].message.content
                    
                    # Clean and format the response
                    formatted_response = format_bot_response(bot_message)
                    
                    # Save bot message
                    bot_msg = Message.objects.create(
                        user=request.user,
                        content=formatted_response,
                        is_bot=True
                    )
                    
                    return JsonResponse({
                        'status': 'success',
                        'message': formatted_response,
                        'user_message': user_message
                    })
                except Exception as e:
                    # Log detailed non-streaming error
                    logger.error(f"Non-streaming error with model {current_model}: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Error details: {str(e)}")
                    raise  # Re-raise to be caught by outer try block
                
            except Exception as e:
                logger.error(f"Error getting response from model: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error details: {str(e)}")
                
                error_message = f"I apologize, but I'm having trouble processing your request right now. Please try again with a shorter message or try later. Error: {str(e)}"
                error_msg = Message.objects.create(
                    user=request.user,
                    content=error_message,
                    is_bot=True
                )
                return JsonResponse({
                    'status': 'error',
                    'message': error_message
                })
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid request format'
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

# Function to extract keywords from user message
def extract_keywords(text):
    # Remove common stop words
    stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'about', 'is', 'am', 'are']
    
    # Tokenize and filter
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if len(word) > 3 and word not in stop_words]
    
    # Return unique keywords
    return list(set(keywords))

@login_required
def settings_dashboard(request):
    # Get or create user preferences
    preference, created = UserPreference.objects.get_or_create(user=request.user)
    
    # Process model preference form
    if request.method == 'POST':
        # Check if it's an AJAX request
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        # Handle form submissions
        form_type = request.POST.get('form_type', '')
        
        if 'preferred_model' in request.POST or form_type == 'model_selection':
            # Handle model selection form - only update the model field
            preferred_model = request.POST.get('preferred_model')
            if preferred_model in dict(UserPreference.MODEL_CHOICES):
                # Update only the model preference
                preference.preferred_model = preferred_model
                preference.save(update_fields=['preferred_model'])
                
                if is_ajax:
                    return JsonResponse({
                        'status': 'success',
                        'message': 'AI model preferences updated successfully!'
                    })
                else:
                    django_messages.success(request, 'AI model preferences updated successfully!')
                    return redirect('settings_dashboard')
            elif is_ajax:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid model selection'
                })
        
        elif 'use_resources' in request.POST or form_type == 'resource_settings':
            # Handle resource settings form
            use_resources = request.POST.get('use_resources') == 'on'
            
            # Update preference
            preference.use_resources = use_resources
            preference.save()
            
            if is_ajax:
                return JsonResponse({
                    'status': 'success',
                    'message': 'Resource settings saved successfully!'
                })
            else:
                django_messages.success(request, 'Resource settings saved successfully!')
                return redirect('settings_dashboard')
        
        elif 'use_custom_prompt' in request.POST or form_type == 'prompt_engineering':
            # Handle prompt engineering form
            use_custom_prompt = request.POST.get('use_custom_prompt') == 'on'
            custom_system_prompt = request.POST.get('custom_system_prompt', '')
            
            # Check if prompt has changed
            prompt_changed = (preference.use_custom_prompt != use_custom_prompt or 
                              preference.custom_system_prompt != custom_system_prompt)
            
            # Update preferences
            preference.use_custom_prompt = use_custom_prompt
            preference.custom_system_prompt = custom_system_prompt
            
            # Clear cached site name if prompt has changed
            if prompt_changed:
                preference.cached_site_name = None
                preference.site_name_updated_at = None
            
            preference.save()
            
            if is_ajax:
                return JsonResponse({
                    'status': 'success',
                    'message': 'Prompt engineering settings saved successfully!'
                })
            else:
                django_messages.success(request, 'Prompt engineering settings saved successfully!')
                return redirect('settings_dashboard')
                
        # Handle file upload
        elif request.FILES.get('file'):
            uploaded_file = request.FILES['file']
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Create the file record
            new_file = UploadedFile(
                name=uploaded_file.name,
                file=uploaded_file,
                size=uploaded_file.size,
                user=request.user,
                file_type=file_type
            )
            new_file.save()
            
            # Process the file content in the background
            try:
                process_file_content_async(new_file.id)
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                
            django_messages.success(request, 'File uploaded successfully!')
            return redirect('settings_dashboard')

    # Create forms
    preference_form = UserPreferenceForm(instance=preference)
    
    # Get all uploaded files for this user
    files = UploadedFile.objects.filter(user=request.user)
    
    # Add current parameter values to context
    context = {
        'files': files,
        'preference_form': preference_form,
        'current_params': {
            'temperature': preference.temperature,
            'max_tokens': preference.max_tokens,
            'stream': preference.stream,
            'use_custom_prompt': preference.use_custom_prompt,
            'custom_system_prompt': preference.custom_system_prompt,
            'use_resources': preference.use_resources
        }
    }
    return render(request, 'admin/dashboard.html', context)

@login_required
def delete_file(request, file_id):
    if request.method == 'POST':
        try:
            file = UploadedFile.objects.get(id=file_id)
            file.file.delete()  # Delete the actual file
            file.delete()  # Delete the database record
            django_messages.success(request, 'File deleted successfully!')
        except UploadedFile.DoesNotExist:
            django_messages.error(request, 'File not found!')
    return redirect('settings_dashboard')

def test_view(request):
    """A simple test view to verify the server is working"""
    return JsonResponse({'status': 'success', 'message': 'Server is working correctly'})

@login_required
@csrf_exempt
def test_send(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            
            if user_message:
                # Save user message
                user_msg = Message.objects.create(
                    user=request.user,
                    content=user_message,
                    is_bot=False
                )
                logger.info(f"Test function: Saved user message with ID {user_msg.id}")
                
                # Create hardcoded response without using AI
                bot_message = f"This is a test response. You said: '{user_message}'"
                
                # Save bot response
                bot_msg = Message.objects.create(
                    user=request.user,
                    content=bot_message,
                    is_bot=True
                )
                logger.info(f"Test function: Saved bot response with ID {bot_msg.id}")
                
                return JsonResponse({
                    'status': 'success',
                    'message': bot_message,
                    'user_message': user_message,
                    'model_used': 'test_model'
                })
            
            return JsonResponse({'status': 'error', 'message': 'Message cannot be empty'})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@login_required
@csrf_exempt
def update_model_params(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Get or create user preferences
            preference, created = UserPreference.objects.get_or_create(user=request.user)
            
            # Update parameters
            preference.temperature = float(data.get('temperature', 0.6))
            preference.max_tokens = int(data.get('max_tokens', 4096))
            preference.stream = bool(data.get('stream', True))
            
            # Validate parameters
            if not (0 <= preference.temperature <= 1):
                return JsonResponse({
                    'status': 'error',
                    'message': 'Temperature must be between 0 and 1'
                })
            
            if not (1000 <= preference.max_tokens <= 32767):
                return JsonResponse({
                    'status': 'error',
                    'message': 'Max tokens must be between 1000 and 32767'
                })
            
            # Save changes
            preference.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Parameters updated successfully'
            })
            
        except Exception as e:
            logger.error(f"Error updating model parameters: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@login_required
@csrf_exempt
def get_site_name(request):
    """Get a dynamic site name based on the user's custom system prompt."""
    try:
        # Get user preferences
        preference, created = UserPreference.objects.get_or_create(user=request.user)
        
        # Default site name
        default_site_name = "Mental Health Counselor"
        
        # Check for force refresh parameter
        force_refresh = request.GET.get('force', 'false').lower() == 'true'
        
        # If not using custom prompt, return default
        if not preference.use_custom_prompt or not preference.custom_system_prompt:
            # If there was a cached name but now we're not using custom prompt, clear it
            if preference.cached_site_name:
                preference.cached_site_name = None
                preference.site_name_updated_at = None
                preference.save(update_fields=['cached_site_name', 'site_name_updated_at'])
                
            return JsonResponse({
                'status': 'success',
                'site_name': default_site_name
            })
        
        # Check if we already have a valid cached site name and not forcing refresh
        if not force_refresh and preference.cached_site_name and preference.site_name_updated_at:
            # Check if cache is less than 1 hour old
            cache_age = timezone.now() - preference.site_name_updated_at
            if cache_age.total_seconds() < 3600:  # 1 hour in seconds
                return JsonResponse({
                    'status': 'success',
                    'site_name': preference.cached_site_name
                })
        
        # If we get here, we need to generate a new site name
        try:
            client = Groq(api_key=groq_api_key)
            
            # Create a prompt to generate a site name
            system_prompt = "You are a helpful naming assistant. Generate a short, creative name for a counseling website based on the provided system prompt. The name should be 2-4 words maximum and reflect the essence of the counseling service described."
            user_prompt = (
                f"Based on the following system prompt for an AI mental health counselor, generate a short, "
                f"creative name for the counseling website (2-4 words maximum). "
                f"Return ONLY the name, with no quotes or explanations:\n\n{preference.custom_system_prompt}"
            )
            
            # Make the API request
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Use fast model for name generation
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=20
            )
            
            # Get the generated name
            site_name = response.choices[0].message.content.strip()
            
            # Clean up the name
            site_name = re.sub(r'["\']', '', site_name)  # Remove quotes
            site_name = re.sub(r'\s+', ' ', site_name)   # Normalize spaces
            site_name = site_name.strip()
            
            # Ensure name is not too long
            words = site_name.split()
            if len(words) > 4:
                site_name = " ".join(words[:4])
            
            # Cache the generated name
            preference.cached_site_name = site_name
            preference.site_name_updated_at = timezone.now()
            preference.save(update_fields=['cached_site_name', 'site_name_updated_at'])
            
            return JsonResponse({
                'status': 'success',
                'site_name': site_name
            })
            
        except Exception as e:
            logger.error(f"Error generating site name: {str(e)}")
            # Clear cache on error
            preference.cached_site_name = None
            preference.site_name_updated_at = None
            preference.save(update_fields=['cached_site_name', 'site_name_updated_at'])
            
            return JsonResponse({
                'status': 'error',
                'site_name': default_site_name
            })
            
    except Exception as e:
        logger.error(f"Error in get_site_name: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'site_name': "Mental Health Counselor"
        })

@login_required
@csrf_exempt
def notify_prompt_update(request):
    """Notify that prompt engineering settings have been updated.
    This endpoint is called after prompt engineering settings are saved,
    and returns a timestamp that can be used by clients to refresh their state.
    """
    # Get user preferences to send along with the notification
    preference, created = UserPreference.objects.get_or_create(user=request.user)
    
    # Clear any cached suggestions for this user to force regeneration
    use_custom = preference.use_custom_prompt and preference.custom_system_prompt
    user_id = request.user.id
    
    if use_custom:
        # Clear custom prompt cache
        prompt_hash = hash(preference.custom_system_prompt)
        cache_key = f"suggestions_cache_user_{user_id}_custom_{prompt_hash}"
    else:
        # Clear default prompt cache
        cache_key = f"suggestions_cache_user_{user_id}_default"
        
    # Delete the cache entry to force regeneration
    cache.delete(cache_key)
    
    return JsonResponse({
        'status': 'success',
        'timestamp': timezone.now().timestamp(),
        'use_custom_prompt': preference.use_custom_prompt,
        'message': 'Prompt update notification sent successfully'
    })

# Function to process file content
def process_file_content(file_id):
    try:
        uploaded_file = UploadedFile.objects.get(id=file_id)
        
        # Skip if already processed
        if uploaded_file.processed and uploaded_file.content:
            return
        
        file_path = uploaded_file.file.path
        file_extension = uploaded_file.file_type or file_path.split('.')[-1].lower()
        
        # Extract text based on file type
        content = ""
        
        if file_extension == 'txt':
            # Process plain text files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
        elif file_extension in ['docx', 'doc']:
            # Process Word documents
            try:
                import docx
                doc = docx.Document(file_path)
                content = "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            except ImportError:
                logger.error("python-docx library not installed for processing Word documents")
                content = "Error: Could not process Word document. Missing python-docx library."
                
        elif file_extension == 'pdf':
            # Process PDF files
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
            except ImportError:
                logger.error("PyPDF2 library not installed for processing PDF files")
                content = "Error: Could not process PDF file. Missing PyPDF2 library."
                
        elif file_extension in ['csv', 'tsv']:
            # Process CSV/TSV files
            try:
                import csv
                delimiter = ',' if file_extension == 'csv' else '\t'
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    csv_reader = csv.reader(f, delimiter=delimiter)
                    rows = list(csv_reader)
                    # For CSV files, create a text representation
                    if len(rows) > 0:
                        headers = rows[0] if len(rows) > 0 else []
                        content = "CSV Content:\n\n"
                        for i, row in enumerate(rows):
                            if i == 0:
                                content += "Headers: " + ", ".join(row) + "\n\n"
                            else:
                                content += f"Row {i}: " + ", ".join(row) + "\n"
            except Exception as e:
                logger.error(f"Error processing CSV/TSV file: {str(e)}")
                content = f"Error: Could not process CSV/TSV file. {str(e)}"
        
        # Clean up content
        if content:
            # Remove excessive whitespace and normalize line breaks
            content = re.sub(r'\n{3,}', '\n\n', content)
            content = re.sub(r'\s{2,}', ' ', content)
            content = content.strip()
        
        # For very large content, truncate or summarize
        if len(content) > 50000:
            content = content[:50000] + "...\n[Content truncated due to size]"
            
            # Optionally create a summary using the LLM
            try:
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",  # Use the smaller model for summarization
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes document content."},
                        {"role": "user", "content": f"Please summarize the following document content. Extract key information, main points, and important details:\n\n{content[:30000]}"}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                summary = response.choices[0].message.content
                content = content[:5000] + "\n\n--- DOCUMENT SUMMARY ---\n\n" + summary
            except Exception as e:
                logger.error(f"Error generating summary for large document: {str(e)}")
        
        # Save processed content to the database
        uploaded_file.content = content
        uploaded_file.processed = True
        uploaded_file.save()
        
        logger.info(f"Successfully processed file {uploaded_file.name} with ID {file_id}")
        
    except UploadedFile.DoesNotExist:
        logger.error(f"File with ID {file_id} does not exist")
    except Exception as e:
        logger.error(f"Error processing file content: {str(e)}")

# Process file in a background thread to not block the request
def process_file_content_async(file_id):
    thread = threading.Thread(target=process_file_content, args=(file_id,))
    thread.daemon = True
    thread.start()

# Function to format bot responses for better readability
def format_bot_response(text):
    """Format the bot's response with proper HTML for better display in the browser."""
    if not text:
        return text
    
    # Convert Markdown style formatting to HTML
    
    # Handle paragraphs - ensure double newlines become paragraph breaks
    paragraphs = text.split('\n\n')
    text = ''.join([f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()])
    
    # Convert Bold - **text** to <strong>text</strong>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    
    # Convert Italic - *text* to <em>text</em>
    text = re.sub(r'\*([^*<>]+)\*', r'<em>\1</em>', text)
    
    # Process lists
    # First convert the paragraph with list items into a proper list structure
    def process_lists(match):
        list_text = match.group(1)
        items = re.findall(r'^\s*\*\s+(.+)$', list_text, re.MULTILINE)
        if items:
            list_html = '<ul>\n'
            for item in items:
                list_html += f'  <li>{item}</li>\n'
            list_html += '</ul>'
            return list_html
        return match.group(0)
    
    # Find paragraphs containing list items and convert them
    text = re.sub(r'<p>((?:\s*\*\s+.+\n?)+)</p>', process_lists, text)
    
    # Process numbered lists or + bulletpoints
    def process_plus_or_numbered_lists(match):
        list_text = match.group(1)
        # Check if it's + bullet points or numbered list (1., 2., etc.)
        items = re.findall(r'^\s*\+\s+(.+)$', list_text, re.MULTILINE)
        if not items:
            items = re.findall(r'^\s*\d+\.\s+(.+)$', list_text, re.MULTILINE)
        
        if items:
            list_html = '<ul>\n'
            for item in items:
                list_html += f'  <li>{item}</li>\n'
            list_html += '</ul>'
            return list_html
        return match.group(0)
    
    # Find paragraphs containing + bulletpoints or numbered lists and convert them
    text = re.sub(r'<p>((?:\s*[\+\d\.]\s+.+\n?)+)</p>', process_plus_or_numbered_lists, text)
    
    # Clean up any lingering <br> tags inside list items
    text = re.sub(r'<li>(.*?)<br>(.*?)</li>', r'<li>\1 \2</li>', text)
    
    # Fix any double paragraph tags
    text = text.replace('<p><p>', '<p>').replace('</p></p>', '</p>')
    
    # Fix line breaks in paragraphs - single newlines become <br>
    text = re.sub(r'([^>])\n([^<])', r'\1<br>\2', text)
    
    return text