# Integration Guide

This guide provides comprehensive examples for integrating Fluid Server with various programming languages and frameworks. Fluid Server provides OpenAI-compatible APIs, making it a drop-in replacement for OpenAI services in your applications.

## API Endpoints Overview

- **Base URL**: `http://localhost:8080/v1`
- **Health Check**: `http://localhost:8080/health`
- **API Documentation**: `http://localhost:8080/docs`

### Core Endpoints
- `POST /v1/chat/completions` - Chat completions with streaming support
- `POST /v1/audio/transcriptions` - Audio transcription
- `POST /v1/embeddings` - Text embeddings generation
- `GET /v1/models` - List available models

## Command Line Testing (curl)

### Health Check
```bash
curl http://localhost:8080/health
```

### Chat Completion (Non-streaming)
```powershell
curl -X POST http://localhost:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{
    "model": "qwen3-8b-int8-ov", 
    "messages": [{"role": "user", "content": "Hello!"}], 
    "max_tokens": 100
  }'
```

### Chat Completion (Streaming)
```powershell
curl -X POST http://localhost:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{
    "model": "qwen3-8b-int8-ov", 
    "messages": [{"role": "user", "content": "Tell me a story"}], 
    "stream": true,
    "max_tokens": 200
  }'
```

### Audio Transcription
```powershell
# Using QNN model (Snapdragon)
curl -X POST http://localhost:8080/v1/audio/transcriptions `
  -F "file=@audio.wav" `
  -F "model=whisper-large-v3-turbo-qnn" `
  -F "response_format=json"

# Using OpenVINO model (Intel)
curl -X POST http://localhost:8080/v1/audio/transcriptions `
  -F "file=@audio.wav" `
  -F "model=whisper-large-v3-turbo-ov-npu" `
  -F "response_format=verbose_json"
```

### Text Embeddings
```powershell
curl -X POST http://localhost:8080/v1/embeddings `
  -H "Content-Type: application/json" `
  -d '{
    "input": ["Hello world", "Vector database"], 
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }'
```

### List Models
```bash
curl http://localhost:8080/v1/models
```

## Python Integration

### Using OpenAI SDK

Install the OpenAI Python SDK:
```bash
pip install openai
```

#### Basic Setup
```python
from openai import OpenAI

# Point to local Fluid Server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local"  # Can be anything for local server
)
```

#### Chat Completions
```python
# Non-streaming completion
response = client.chat.completions.create(
    model="qwen3-8b-int8-ov",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].message.content)
```

#### Streaming Chat Completions
```python
# Streaming completion
response = client.chat.completions.create(
    model="qwen3-8b-int8-ov",
    messages=[{"role": "user", "content": "Write a short poem about AI"}],
    stream=True,
    max_tokens=100
)

print("AI Response:")
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # New line at end
```

#### Audio Transcription
```python
# Transcribe audio file
with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo-qnn",  # or whisper-large-v3-turbo-ov-npu
        file=audio_file,
        response_format="verbose_json"
    )

print(f"Transcribed text: {transcript.text}")
print(f"Language: {transcript.language}")
print(f"Duration: {transcript.duration}s")
```

#### Text Embeddings
```python
# Generate embeddings
embeddings = client.embeddings.create(
    model="sentence-transformers/all-MiniLM-L6-v2",
    input=["Text to embed", "Another piece of text", "Vector search query"]
)

for i, embedding in enumerate(embeddings.data):
    print(f"Text {i+1} embedding dimensions: {len(embedding.embedding)}")
    print(f"First 5 values: {embedding.embedding[:5]}")
```

### Error Handling
```python
from openai import OpenAI, APIError, APIConnectionError

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

try:
    response = client.chat.completions.create(
        model="qwen3-8b-int8-ov",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
    
except APIConnectionError:
    print("Failed to connect to Fluid Server. Is it running?")
except APIError as e:
    print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## .NET Integration

### Using OpenAI SDK for .NET

Install the Azure OpenAI SDK for .NET:
```xml
<PackageReference Include="Azure.AI.OpenAI" Version="1.0.0-beta.17" />
```

#### Basic Setup
```csharp
using Azure.AI.OpenAI;
using Azure;

var client = new OpenAIClient(
    new Uri("http://localhost:8080/v1"),
    new AzureKeyCredential("local")  // Can be anything for local server
);
```

#### Chat Completions
```csharp
var chatOptions = new ChatCompletionsOptions()
{
    DeploymentName = "qwen3-8b-int8-ov",
    Messages = {
        new ChatRequestSystemMessage("You are a helpful assistant."),
        new ChatRequestUserMessage("Explain machine learning briefly.")
    },
    MaxTokens = 150,
    Temperature = 0.7f
};

var response = await client.GetChatCompletionsAsync(chatOptions);
Console.WriteLine(response.Value.Choices[0].Message.Content);
```

#### Streaming Chat Completions
```csharp
var chatOptions = new ChatCompletionsOptions()
{
    DeploymentName = "qwen3-8b-int8-ov", 
    Messages = { new ChatRequestUserMessage("Write a haiku about programming") },
    MaxTokens = 100
};

await foreach (var choice in client.GetChatCompletionsStreaming(chatOptions))
{
    if (choice.ContentUpdate != null)
    {
        Console.Write(choice.ContentUpdate);
    }
}
Console.WriteLine();
```

#### Audio Transcription
```csharp
using var audioStream = File.OpenRead("audio.wav");

var transcriptionOptions = new AudioTranscriptionOptions()
{
    DeploymentName = "whisper-large-v3-turbo-qnn",
    AudioData = BinaryData.FromStream(audioStream),
    ResponseFormat = AudioTranscriptionFormat.VerboseJson
};

var transcription = await client.GetAudioTranscriptionAsync(transcriptionOptions);
Console.WriteLine($"Transcribed: {transcription.Value.Text}");
Console.WriteLine($"Language: {transcription.Value.Language}");
```

## Node.js Integration

### Using OpenAI SDK for Node.js

Install the OpenAI Node.js SDK:
```bash
npm install openai
```

#### Basic Setup
```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
    baseURL: 'http://localhost:8080/v1',
    apiKey: 'local', // Can be anything for local server
});
```

#### Chat Completions
```javascript
async function chatCompletion() {
    try {
        const completion = await openai.chat.completions.create({
            model: 'qwen3-8b-int8-ov',
            messages: [
                { role: 'system', content: 'You are a helpful assistant.' },
                { role: 'user', content: 'Explain async/await in JavaScript' }
            ],
            max_tokens: 200,
            temperature: 0.7
        });

        console.log(completion.choices[0].message.content);
    } catch (error) {
        console.error('Chat completion error:', error);
    }
}
```

#### Streaming Chat Completions
```javascript
async function streamingChat() {
    try {
        const stream = await openai.chat.completions.create({
            model: 'qwen3-8b-int8-ov',
            messages: [{ role: 'user', content: 'Tell me about Node.js' }],
            stream: true,
            max_tokens: 150
        });

        for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
                process.stdout.write(content);
            }
        }
        console.log(); // New line
    } catch (error) {
        console.error('Streaming error:', error);
    }
}
```

#### Audio Transcription
```javascript
import fs from 'fs';

async function transcribeAudio() {
    try {
        const transcription = await openai.audio.transcriptions.create({
            file: fs.createReadStream('audio.wav'),
            model: 'whisper-large-v3-turbo-qnn',
            response_format: 'verbose_json'
        });

        console.log('Transcription:', transcription.text);
        console.log('Language:', transcription.language);
        console.log('Duration:', transcription.duration);
    } catch (error) {
        console.error('Transcription error:', error);
    }
}
```

## Best Practices

### Connection Management
- Use connection pooling for high-throughput applications
- Implement proper retry logic with exponential backoff
- Monitor connection health and implement graceful degradation

### Performance Optimization
- Use streaming for long responses to improve perceived performance
- Batch multiple requests when possible
- Consider model warm-up time for the first request

### Error Handling
- Implement comprehensive error handling for network issues
- Handle model loading delays during server startup
- Provide fallback mechanisms for service unavailability

### Security Considerations
- Run Fluid Server on localhost for development
- Use proper network security for production deployments
- Validate all inputs before sending to the API

### Model Selection
- Choose appropriate models based on your hardware capabilities
- Consider the trade-off between model size and performance
- Test different models to find the best fit for your use case