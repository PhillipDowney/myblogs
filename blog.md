# Unleashing Potential with Audio-to-Speech Translation
## A Key to Seamless Connectivity and Accessibility 

In our rapidly evolving digital landscape, seamless communication is essential, and the capacity for precise audio transcription has emerged as a cornerstone requirement. This necessity is especially vital in sectors such as healthcare, emergency services, and business productivity tools where interpreting spoken words accurately can often determine life-or-death outcomes or significantly enhance efficiency.
**Possible Applications:**
1.  **Emergency Services Calls:** Transcribing audio from emergency calls offers crucial data for first responders, potentially improving response times and accuracy. This technology also streamlines the creation of reports, lessening manual transcription efforts.
2.  **Productivity Enhancement:** In professional environments, meetings frequently result in discussions needing documentation. Audio-to-text conversion can automate this process, freeing valuable time for more productive tasks.
3.  **Assisting the Hearing Impaired:** For individuals with hearing impairments, converting audio to text greatly improves accessibility, enabling them to engage more easily in conversations and understand spoken content.
4.  **Narration:** This feature can significantly reduce time and costs associated with creating voiceovers for videos, podcasts, or other digital media.
5.  **Call Centers:** Transcribing call centre activities allows for valuable insights to be extracted, aiding in improving future customer engagement strategies.
6.  **Voice-Enabled Applications:** With advancements in AI, speech is becoming an essential input and output method for various applications, enhancing agent integration.
<br>

### Understanding how to implement a simple transcription service
For this example we are using a combination of IBM Granite-Speech Large Language models, VLLM and python to implement a simple transcriber, using all open soure components

Granite speech is a compact and efficient speech-language model, built on top of IBMs Granite language model and specifically designed for English automatic speech recognition (ASR). It is available in a 2b and 8b sized model, both providing outstanding accuracy in transcribing the most challenging of audio files in real life scenarios.

vLLM is a fast and easy-to-use library for LLM inference and serving, and using audio extensions becomes a powerful server for transcribing audio data that is streamted to it through a simple LLM interface.

We will use a simple python script to perform the translation, in this script we use the OpenAI open source client which vLLM supports as API standard.<br>


![illustration](assets/audio-1.svg)

From the below diagram we are implementing the vLLM container with IBM Granite Speech and the transcriber component.In further blogs we will show how you can breakup ans stream audio in chunks and use request information in the audio to know where to route the information to. For this example we are taking a simple input/ output approach that will work for wav or mp3 formats.<br>

### Building Your Server <br>
Deploying your server for audio-to-speech translation involves constructing a Virtual Large Language Model (VLLM) using the vllm-openai Docker image available on Docker Hub (https://hub.docker.com/r/vllm/vllm-openai/tags). The source code can be accessed here: https://github.com/vllm-project/vllm <br>

<br>

#### Step1: Create Docker File 

<br>

Create a `Dockerfile` as follows: <br>
```Dockerfile
FROM vllm/vllm-openai:latest
RUN pip install --system librosa soundfile mistral_common[audio]
```
This Dockerfile uses the vllm-openai base image and installs necessary audio processing libraries.

<br>

####  Step 2: Build your image 

<br>

Build your image with the following command. you can exchange `podman` for `docker` if you are using docker <br>
```bash
podman build -t customvllm .
```
<br>

#### Step 3: Deployment on Podman or Docker 

<br>

To run this instance, you need to pull the image and initiate a container using either Podman or Docker. Here's an example command with GPU access (ensuring nvidia-docker is installed): <br>

you will need a X86 enviironment to run this. As per below we have this running on Windows environment using Podman. It should run on Windows or Linu using docker or podman container solutions. <br>
This demonstration ran on a system running:
- Windows 11 
- i9--12900k intel processor 
- Nvidia-Gforce 4060 TI

The demonstration used less than 18GB of memory and 6GB of GPU memory and 10% of CPU, however when starting the container you will notice it will use as much GPU memory as possible but then reduce once ready to around 6GB. <br>



```bash
podman run --device nvidia.com/gpu=all -v C:\huggingface:/root/.cache/huggingface -p 8000:8000 --env "HUGGING_FACE_HUB_TOKEN=<token>" customvllm --model "ibm-granite/granite-speech-3.3-2b" --max-model-len "2048" --enable-lora --lora-modules "speech=ibm-granite/granite-speech-3.3-2b"  --max-lora-rank "64" --quantization fp8 
```

<br>

***Note*** just swap 2b for 8b in the name of the model if you are running on a higher spec GPU. <br>

<br>
Replace `YOUR_TOKEN`  with your HuggingFace token for model access, which is optional if you're behind a firewall. This command downloads the IBM Granite Speech model from Hugging Face and sets it up within the container for speech transcription.<br>

The model will take a few minutes to come up as it loads 


### Accessing Your Model <br>

To transcribe the following voice transmission using the model the following code using the OpenAI client will take as an argument a given .wav or .mp3 file and return the transcription, and in addition print out a summary of the conversation.
The `api_base` is set to the local host but can also be set by exporting an environment variable `API_BASE` you can set the base url to a different endpoint, like wide if it sitting behind a proxy you can export the variable `API_KEY` to set a api token to get through the proxy.<br>
No changes required if you are running locally.

<br>

```python
import base64
from openai import OpenAI
import sys, os
api_base = "http://127.0.0.1:8000"
api_key = "<APIKEY>".   # Note: only needed if accessing through a proxy.

client = OpenAI(
    # defaults to os.environ.get("API_KEY")
    api_key=os.environ.get("API_KEY", oapi_key),
    base_url=os.environ.get("API_BASE", api_base),
    timeout=12000,
)
base_model_name = "ibm-granite/granite-speech-3.3-2b"
lora_model_name = "speech"
def transcribe(file_location="./pilot_voice.wav"):
    if not os.path.exists(file_location):
        return "Error: File cannot be located"
    audio_base64 = base64.b64encode(open(file_location, "rb").read()).decode("utf-8")
    question = "What is being said in this audio?"
    chat_completion_with_audio = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_base64, "format": "wav"},  # Specify the format of the audio
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=64,
        stream=False,
        model = lora_model_name,
    )
    print(f"Audio Example - Question: {question}")
    """
    for chunk in chat_completion_with_audio:
        content = chunk.choices[0].delta.content or ""
        partial_text += content
        # Here, you would send 'content' or accumulated 'partial_text'
        # to your chosen streaming TTS model/service.
        # For example:
        # synthesize_and_play_audio(content)
        print(f"Received text chunk: '{content}'")  # For demonstration
    """
    try:
        print("----------------------------------------------------------------------")
        response = client.chat.completions.create(
            model=base_model_name,
            messages=[
                {
                    "role": "user",
                    "content": 'Summarise the following for me by action in point form """'
                    + chat_completion_with_audio.choices[0].message.content
                    + '"""',
                }
            ],
        )
        print(response.choices[0].message.content)
        return chat_completion_with_audio.choices[0].message.content
    except Exception as e:
        print(e)
        return "Error Transcribing audio"
if __name__ == "__main__":
    file = sys.argv[1]
    transcribe(file)
```

<br>

Save the above python code to a`audio_transcribe.py` in your local directory and download the folowing audio file `link to be provided` and copy to the samer directory. <br>

Then install the `openai` python library on python 3.11 using: <br>

```python
pip install openai
```

<br>

To execute this script, use the command:<br>

```bash
python3 audio_transcribe.py pilot_voice.wav
```

[Download the .wav file here](https://github.com/PhillipDowney/myblogs/blob/d936a24383e7254381e63ccb1110ff2de3fe1a46/assets/pilots_voice.wav)
<br>

### Results <br>
```
length of audio file : 1268896
Audio Example - Question: What is being said in this audio?
----------------------------------------------------------------------
Spoken text: we are at cruise altitude on the flight this is the captain speaking i'm on the flight deck on the left side of the aircraft i have my noise cancelling headset on but the microphone is away from my mouth and it's difficult for me to hear anything apart from the announcements
----------------------------------------------------------------------
1. Aircraft is at cruising altitude.
2. Pilot (Captain) giving an announcement.
3. Pilot is seated on the left side of the flight deck.
4. Pilot is using noise-cancelling headset.
5. The microphone of the headset is positioned away from the pilot's mouth.
6. Pilot encounters difficulty hearing due to headset position,save for in-flight announcements.
```

### Conclusion <br>

In essence, Audio-to-Speech translation using Granite Speech combined with vLLM is a powerful tool that enhances connectivity, amplifies productivity, and improves accessibility for individuals with hearing impairments. By utilizing Containers and GPU acceleration, you can easily establish an efficient system capable of managing diverse audio transcription tasks effectively.

This demonstrates how easily, with minimal resources, a transcribing application using AI can be implemented that can overcome the most onerous of audio recordings.



