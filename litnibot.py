#!/usr/bin/env python
# coding: utf-8

# In[1]:


from telethon import TelegramClient, events
from telethon.tl.types import UpdateNewMessage
import json
import re
import openai
import math
import asyncio
import random
import os, sys
import subprocess
from PIL import Image
import cv2
import numpy as np
from pydub import AudioSegment

from googleapiclient import discovery
import httplib2
from oauth2client.client import GoogleCredentials


# In[2]:


from bot_config import *
MAX_RETRYS = 3


# In[3]:


print ("libs imported!")


# In[4]:


SEC_PER_VID_MB = 10
NOISE_FILE = 'pink.wav'
WORKDIR = 'workdir'


# In[5]:


def non_english_symbols_regex(string: str):
    pattern = re.compile(r'[^\x20-\x7E]')
    return pattern.findall(string)


# In[6]:


def estimate_token_count(prompt):
    estimated_token_count = 0
    words = prompt.split()
    for word in words:
        if len(non_english_symbols_regex(word)) > 1:
            estimated_token_count+=TOKENS_PER_WORD['rus']
        else:
            estimated_token_count+=TOKENS_PER_WORD['eng']
    estimated_token_count = int(estimated_token_count)
    return int(estimated_token_count)


# In[7]:


def get_tanslate_service():
    credentials = GoogleCredentials.get_application_default()
#    credentials = GoogleCredentials.get_application_default().create_scoped ("https://www.googleapis.com/auth/cloud-platform")
    http = httplib2.Http()
    credentials.authorize(http)
    
    creds = GoogleCredentials.get_application_default()
    creds.authorize(http)
    # Create a service object
    translate_service = discovery.build('translate', 'v2')#, http=http)
    
    # Create a service object
#    service = discovery.build('translate', 'v3', http=http, discoveryServiceUrl=DISCOVERY_URL)
    return translate_service


# In[8]:


def translate_to_lang(texts, target_lang):
    response = translate_service.translations().list(q=texts,target=target_lang).execute()
    return [i['translatedText'] for i in response['translations']][0]


# In[9]:


def extract_audio(in_video_path, out_audio_path):
    try:
        aud = AudioSegment.from_file(in_video_path, "mp4")
        aud.export(out_audio_path, format="mp3")
        print (f"Extracted audio to: {out_audio_path}")
        return out_audio_path
    except Exception as e:
        aud_file = None
        print (f"Could NOT extract audio: {str(e)}")
        return None


# In[10]:


def merge_media(vide_file, audio_file, res_file):
    command = [
        "ffmpeg",
        "-i", vide_file,
        "-i", audio_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        "-y",
        res_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res_file
    except subprocess.CalledProcessError as error:
        print("Error: FFmpeg command failed with return code", error.returncode)
        print("Error output:", error.stderr.decode(), file=sys.stderr)
        return vide_file
    


# In[11]:


async def rewrite_text (rules, text):
    token_used = 0
    prompt = rules + '\n\n' + text + "###"
    max_tokens = MAX_REQUEST_LENGHT - CONTIGENCY - estimate_token_count(prompt)
    retrys = 0
    while retrys < MAX_RETRYS:
        retrys+=1
        try:
            resp = await openai.Completion.acreate(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=max_tokens,
                n=1,
                stop='###',
                temperature=TEMPERATURE,
                timeout = 15
            )
            token_used = resp["usage"]["total_tokens"] 
            answer = str(resp.choices[0].text).strip()
            break
        except Exception as e:
            answer = f"Got error from OpenAI:\n\n{str(e)}"
            print(answer)
    return answer, token_used


# In[12]:


def rotate_video(in_file, angle_max):
    sep = '\\' if len(in_file.split('\\')) > len(in_file.split('/')) else '/'
    out_file = os.path.join(*in_file.split(sep)[:-1], 'rotated_'+in_file.split(sep)[-1])
    
    try:
        cap = cv2.VideoCapture(in_file)
        width, height, fps= int(cap.get(3)),int(cap.get(4)), (cap.get(5))
        center = (int(width/ 2), int(height/ 2))
        aspect_ratio = width / height
        angle = round((1+ random.random()) * angle_max /2, 5) * random.choice([-1,1])
#        scale = np.sqrt(aspect_ratio * np.cos(np.radians(angle)) ** 2 + np.sin(np.radians(angle)) ** 2) / np.sqrt(aspect_ratio)
        scale = 1.01
        print (f"Rotating video {angle} degrees, scale {scale}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or other codecs, e.g. 'mp4v'
        out = cv2.VideoWriter(out_file, fourcc, int(fps), (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
            frame = cv2.warpAffine(frame, rot_mat, (width, height))
            out.write(frame)

        cap.release()
        out.release()
        print ("Rotation Done!")
        return out_file
    except Exception as e:
        print (f"Could NOT extract audio: {str(e)}")
        return in_file


# In[13]:


def get_vid_params(vidfile):
    cap = cv2.VideoCapture(vidfile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = (cap.get(5))
    cap.release()
    return width, height, fps


# In[14]:


def make_work_wm(wmfile, frame_height, frame_width, scale=0.2, of=10, loc='br'):
    try:
        print (f"Making work_wm from {wmfile} with W {frame_width}, H {frame_height}")
        watermark = cv2.imread(wmfile, cv2.IMREAD_UNCHANGED)
        wm_height, wm_width, _ = watermark.shape
        aspect_ratio = wm_width / wm_height
        minsidelen = min (frame_height, frame_width)
        height, width, channels = watermark.shape
        y_offset,x_offset = of, of
        if loc[0] == 'c':
            scale=0.9
            of =0
        new_width = int(minsidelen * scale * aspect_ratio)
        new_height = int(minsidelen * scale)
        watermark = cv2.resize(watermark, (new_width, new_height), interpolation = cv2.INTER_AREA)

        if loc[0] == 'c':
            x_offset = (frame_width - new_width)//2
            y_offset = (frame_height - new_height)//2
            print ("CENTER")
        else:
            if loc[0] == 'b':
                y_offset = frame_height - new_height - of
            if loc[1] == 'r':
                x_offset = frame_width - new_width - of
        result = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = watermark
        resfile = os.path.join(WORKDIR, f'wm_{str(random.random())[2:6]}.png')
        cv2.imwrite(resfile, result)
        print (f"Made watermark: {resfile}")
        return resfile
    except Exception as e:
        print (f"Could NOT make watermark, using as is: {str(e)}")
        return wmfile



# In[15]:


def add_wm_to_vid(vidfile, wmfile):
    sep = '\\' if len(vidfile.split('\\')) > len(vidfile.split('/')) else '/'
    retfile = os.path.join(*vidfile.split(sep)[:-1], 'wm_'+vidfile.split(sep)[-1])
    
    print (f"Adding WM to video: {wmfile} to {vidfile}, will write to {retfile}")
    command = [
        "ffmpeg",
        "-i", vidfile,
        "-i", wmfile,
        "-filter_complex",
        "overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2",
        "-y",
        retfile
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print (f"Added WM to video: {vidfile} to {wmfile}")
        return retfile
    except subprocess.CalledProcessError as error:
        print("Error: FFmpeg command failed with return code", error.returncode)
        print("Error output:", error.stderr.decode(), file=sys.stderr)
        return vidfile


# In[16]:


def make_some_noise4(in_file, noise):
    sep = '\\' if len(in_file.split('\\')) > len(in_file.split('/')) else '/'
    out_file = os.path.join(*in_file.split(sep)[:-1], 'noise_'+in_file.split(sep)[-1])    
    print (f"Making noise with {noise} db")
    sound = AudioSegment.from_file(in_file, format="mp3")
    noise_sound = AudioSegment.from_file(NOISE_FILE, format="wav")

    repeat_times = (len(sound) + len(noise_sound) - 1) // len(noise_sound)
    
    noise_sound = noise_sound * repeat_times
    noise_sound = noise_sound[:len(sound)]
    noise_sound = noise_sound - 40 + noise
    
    out_sound = sound.overlay(noise_sound, loop=True)

    out_sound.export(out_file, format="mp3")
    return out_file


# In[17]:


def process_video(vidfile, vid_rules):
    w, h, fps = get_vid_params(vidfile)
    sep = '\\' if len(vidfile.split('\\')) > len(vidfile.split('/')) else '/'
    out_file = os.path.join(*vidfile.split(sep)[:-1], 'out_'+vidfile.split(sep)[-1])

    aud_path = os.path.join(*vidfile.split(sep)[:-1], 'audio_'+'.'.join(vidfile.split(sep)[-1].split('.')[:-1])+'.mp3')
    aud_file = extract_audio(vidfile, aud_path)
    if vid_rules.get('angle'):
        vidfile = rotate_video(vidfile, vid_rules['angle'])
    if (vid_rules['watermark']):
        wmloc = vid_rules.get('wm_loc')
        if not wmloc or wmloc not in ['br','bl','tr','tl', 'c']:
            wmloc = 'br'
            print (f"wm_loc incorrect, using default: bottom-right ('br')")
        wmfile = make_work_wm (vid_rules['watermark'], h, w, scale=0.2, of=10, loc=wmloc)
        vidfile = add_wm_to_vid(vidfile, wmfile)
        
    if (vid_rules['noise']):
        aud_file = make_some_noise4 (aud_file, vid_rules['noise'])
    outpath = os.path.join(*vidfile.split(sep)[:-1], 'final_'+vidfile.split(sep)[-1])    
    video = merge_media(vidfile, aud_file, outpath)
    return video


# In[18]:


async def process_photo(photo_path, img_rules):
    print ('processing image, rules: ', img_rules)
    sep = '\\' if len(photo_path.split('\\')) > len(photo_path.split('/')) else '/'
    out_file = os.path.join(*photo_path.split(sep)[:-1], 'out_'+photo_path.split(sep)[-1])
    img = Image.open(photo_path).convert("RGBA")
    width, height = img.size

    if img_rules['rotate']:
        print (f"Rotating by rnd {img_rules['rotate']} degrees")
        img = img.rotate(img_rules['rotate'] * random.random())
        
    if (img_rules['crop'] ):
        print (f"cropping by rnd {img_rules['crop']}%")

        left = round(width * (random.random() * img_rules['crop'] / 100))
        right = round(width * (1-random.random() * img_rules['crop'] / 100))
        top = round(height * (random.random() * img_rules['crop'] / 100))
        bottom = round(height * (1-random.random() * img_rules['crop'] / 100))

        img = img.crop((left, top, right, bottom))

    if img_rules['watermark']:
        
        wmloc = img_rules.get('wm_loc')
        if not wmloc or wmloc not in ['br','bl','tr','tl', 'c']:
            wmloc = 'br'
        wmfile = make_work_wm (img_rules['watermark'], height, width, scale=0.2, of=10, loc=wmloc)
        
        print (f"Adding watermark: {img_rules['watermark']}")        
        watermark = Image.open(wmfile)
        w_width, w_height = watermark.size
#        x = img.width - w_width - 10
#        y = img.height - w_height - 10
        # Add the watermark to the image
#        img.alpha_composite(watermark, (x, y))
        img.alpha_composite(watermark, (0, 0))
        
    img = img.convert("RGB")
    img.save(out_file)
    
    return out_file


# In[19]:


async def process_album(update, rules):
    files = []
    new_text = ""
    wd = f"{update.messages[0].chat_id}_{update.messages[0].id}"
    destination_chat = [_['to'] for _ in channels if _['from']==update.chat_id][0]

    for message in update.messages:
        if message.photo:
            photo_path = os.path.join(WORKDIR, wd, f'input_{str(random.random())[2:5]}.jpg')
            await client.download_media(message.photo, photo_path)
            new_photo = await process_photo(photo_path, rules['image'])
            files.append(new_photo)
        elif message.video:
            vidfile = os.path.join(WORKDIR, wd, f'invideo_{str(random.random())[2:5]}.mp4')
            print (f"\nDownloading video to {vidfile}!")
            await client.download_media(message.video, vidfile)
            fw_file = process_video(vidfile, rules['video'])
            files.append(fw_file)
        else:
            print (f"Some other message type: {message.to_dict()}")

        if message.text.strip():
            add_text = await get_processed_text (message.text, rules['text'])
            new_text += '\n'+add_text

    sent = await client.send_message (destination_chat, file = files, message = new_text.strip()[:1023])
    print ("Album sent:", sent)


# In[20]:


async def album_callback(update: UpdateNewMessage):
    global processing_queue
    upd = update.messages
    if update.chat_id in [_['from'] for _ in channels]:
        rules = [_['rules'] for _ in channels if _['from']==update.chat_id][0]
        fnc = process_album (update, rules)
        processing_queue.append(fnc)
        print (f"Added Album to processing (curr {len(processing_queue)})")


# In[21]:


async def processor():
    print (f"Processor started!")
    global processing_queue
    while True:
        q = processing_queue.copy()
        for proc in q:
            await proc
            await asyncio.sleep(1)
            print ("Processed task")
        for proc in q:
            processing_queue.remove(proc)
        await asyncio.sleep(1)


# In[22]:


# Define the callback function that will be called when a new message arrives
async def message_callback(update: UpdateNewMessage):
    global processing_queue
    message = update.message
    # Forward the message to another chat
    if update.message.chat_id in [_['from'] for _ in channels]:
        if message.grouped_id:
            return
        destination_chat = [_['to'] for _ in channels if _['from']==message.chat_id][0]
        rules = [_['rules'] for _ in channels if _['from']==message.chat_id][0]
        fnc = forward_messages(destination_chat, message, rules)
        processing_queue.append(fnc)
        print (f"Added message to processing (curr {len(processing_queue)})")


# In[23]:


async def forward_messages(destination_chat, message, rules):
    
    if (message.text.strip()):
        new_text = await get_processed_text (message.text, rules['text'])
        if "Got error from OpenAI" in new_text:
            await client.send_message(ADMIN_ID, new_text)
            new_text = message.text.strip()
    else:
        print (f"text will be blank")
        new_text = ""

        
    fw_files = []
    wd = f"{message.chat_id}_{message.id}"
    os.mkdir(os.path.join(WORKDIR, wd))
    if (message.photo):
        photo_path = os.path.join(WORKDIR, wd, f'input_{str(random.random())[2:5]}.jpg')
        await client.download_media(message.photo, photo_path)
        fw_file = await process_photo(photo_path, rules['image'])
        fw_files.append(fw_file)
    elif (message.video):
        vidfile = os.path.join(WORKDIR, wd, f'invideo_{str(random.random())[2:5]}.mp4')
        print (f"\nDownloading video to {vidfile}!")
        await client.download_media(message.video, vidfile)
        fw_file = process_video(vidfile, rules['video'])
        fw_files.append(fw_file)
        
    if fw_files:
        if len(new_text)<=1023:
            print (f"Sending single message with file and text:\n{new_text}")
            sent_msg = await client.send_message(destination_chat, file = fw_files, message=new_text)
        else:
            print (f"Sending message with file and then with text")
            sent_msg = await client.send_file(destination_chat, file = fw_files)
            sent_msg2 = await client.send_message(destination_chat, message=new_text[:4095])
    else:
        if len (new_text)>0:
            print (f"Sending only text")
            sent_msg = await client.send_message(destination_chat, message=new_text[:4095])        

    print (f"Message sent to {destination_chat} msg:\n{sent_msg}")


# In[24]:


async def get_processed_text(text, rules):
    if rules =="":
        print (f"Forwarding text:\n{text}")
        return text
    if text =="":
        print (f"NO text:\n{text}")
        return ""
    
    if rules['translate']:
        print (f"Translating to {rules['translate']}")
        text = translate_to_lang(text,rules['translate'])
    
    ret_text = text
    if rules['prompt']:
        print (f"Rewriting text with TXT-rule: '{rules['prompt']}' :\n{text}")    
        text_list = text_to_chunks(text)

        ret_text = ""
        total_tokens = 0
        for text in text_list:
            new_text, tokens = await rewrite_text(text, rules['prompt'])
            ret_text+= f"\n{new_text}"
            total_tokens += tokens
        print (f"\n\nRewrote using {total_tokens} tokens to: \n {ret_text}")
    
    return ret_text
    


# In[25]:


def text_to_chunks(text):
    MAX_TOKENS = (MAX_REQUEST_LENGHT - CONTIGENCY)//2
    total_count = estimate_token_count(text)
    n_chunks = math.ceil(total_count / MAX_TOKENS)
    max_chunk_token_count = MAX_TOKENS // n_chunks + 50
    temp_string = ""
    temp_count = 0
    text_list = []
    
    for word in re.findall(r"[\w']+|[^\s\w]", text):
        if temp_count + estimate_token_count(word) <= max_chunk_token_count:
            temp_string += word + " "
            temp_count += estimate_token_count(word)
        else:
            text_list.append(temp_string.strip())
            temp_string = word + " "
            temp_count = estimate_token_count(word)
    text_list.append(temp_string.strip())
    
    return text_list


# In[26]:


async def main():
    print ("startded main")
    #await client.start(PHONE)
    try:
        await client.connect()
        print ("Client connected")
        if not await client.is_user_authorized():
            await client.send_code_request(PHONE)
            code = input('Enter the code: ')
            await client.sign_in(PHONE, code)
    except errors.SessionPasswordNeededError:
        password = input('Two-step verification is enabled. Please enter your password: ')
        await client.start(password=password)


    dialogs = await client.get_dialogs ()
    all_dialogs = ""
    for dialog in dialogs:
        all_dialogs += f"\n{dialog.id} \t {dialog.title}"
    with open(f"{PHONE}_dialogs.txt", 'w') as f:
        f.write(all_dialogs)
    
    
    # Register the callback function
    client.add_event_handler(album_callback, events.Album)
    client.add_event_handler(message_callback, events.NewMessage)
    

    print (f"Bot up and running")
    # Start polling
    await client.run_until_disconnected()
    


# In[ ]:





# In[ ]:





# In[27]:


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./ai-with-radix-09328f41ef89.json"
CHANNEL_FILE= 'channels.json'
from channels import *
#with open(CHANNEL_FILE, 'r') as f:
#    channels = json.load(f)
if WORKDIR not in os.listdir():
    os.mkdir(WORKDIR)
    
global processing_queue
processing_queue = []


# In[28]:
openai.api_key = openai_key
client = TelegramClient(PHONE, API_ID, API_HASH)
translate_service = get_tanslate_service()

workers = [main(), processor()]
async def runme (workers):
    await asyncio.gather(*workers)
    
if __name__ == '__main__':
    asyncio.run(runme(workers))