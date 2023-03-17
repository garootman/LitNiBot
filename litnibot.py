#!/usr/bin/env python
# coding: utf-8

# In[1]:


from telethon import TelegramClient, events, errors, functions, types

from telethon.tl.types import InputMediaPoll, Poll, PollAnswer
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
import shutil

from datetime import datetime, timedelta
import tiktoken


# In[2]:


from bot_config import *
MAX_RETRYS = 30
OPENAI_RETRY_TIMEOUT = 10
CLEANUP = True


# In[3]:


print ("libs imported!")


# In[4]:


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./ai-with-radix-09328f41ef89.json"
encoding = tiktoken.get_encoding('gpt2')


# In[5]:


SEC_PER_VID_MB = 10
NOISE_FILE = 'pink.wav'
WORKDIR = 'workdir'
if (CLEANUP):
    try:
        shutil.rmtree(WORKDIR, ignore_errors=False)
    except:
        pass

chat = -1001150050047
question = "Опрос такой-то"
aswer_list = ['Ответ номер один', 'Ответ номер 2']
thread=593
# In[6]:


def read_channels():
    global channels
    with open (f"channels.json", 'r', encoding='utf-8') as f:
        channels = json.load(f)
    return channels
#with open (f"channels.json", 'w', encoding='utf-8') as f:
#    json.dump(channels,f, ensure_ascii=False, indent=4)


# In[7]:


async def get_emotion_response(text):
    emostr, tokens = await process_with_openai2 (PROMPT_TO_CHECK_EMOTION, text)
    emolist = [i for i in emostr if i in EMOJI_REACTION_LIST]
    return emolist


# In[8]:


async def send_poll (chat, question='', answer_list=[], poll=None, thread=None):
    
    if not poll:
        poll=Poll(id=random.randint(1,10**6), question=question,
                  answers=[PollAnswer(v, str.encode(f"{i}")) for i, v in enumerate(answer_list)])    
    try:
        sent = await client(functions.messages.SendMediaRequest(peer=chat, message='poll message', media=poll, reply_to_msg_id=thread))
        return sent.updates[0]
    except Exception as e:
        print (f"Could NOT send Poll to {chat} (тред {thread}) ({question}): {str(e)}")
        return None


# In[9]:


async def send_msg(chat, text="", files=None, poll=None,reply=None, comment=None, thread=None):
    global me
    if not me:
        me = await client.get_me()

    retrys = 0
    chunklen = 1024 if files else 4096
    texts_to_send = ['']
    for l in text.splitlines():
        if len (texts_to_send[-1] +'\n'+ l) <=chunklen:
            texts_to_send[-1]+='\n'+l.strip()
        else:
            if len(l)<=chunklen:
                texts_to_send.append(l.strip())
            else:
                for w in l.split():
                    if len (texts_to_send[-1] +' '+ w) <=chunklen:
                        texts_to_send[-1]+=' '+w
                    else:
                        texts_to_send.append(w)
    texts_to_send = [i.strip() for i in texts_to_send if len(i)>0]
    
    sent = False
    
    
    for txt in texts_to_send:
        while retrys < MAX_RETRYS:
            retrys+=1
            try:
                files_to_send = (list(files) if (txt==texts_to_send[0] and files) else None)
                if not thread:
                    sent = await client.send_message (chat, file = files_to_send , message = txt)
                    if type(sent)==list:
                        sent = sent[0]
                else:
                    sent = await client(functions.messages.SendMessageRequest(
                        peer=chat, 
                        message=txt, 
                        reply_to_msg_id = thread,
                        top_msg_id = (reply if reply else 1),       # <- topic id
                    ))
                    sent = sent.updates[1].message
                    if (files_to_send):
                        sent = await client.send_message (chat, file = files_to_send , message = "", reply_to=sent)
                        if type(sent)==list:
                            sent = sent[0]
                break
                    
            except Exception as e:
                print (f"Error sending message to {str(chat)}: {str(e)}")
                await asyncio.sleep(5)
    
    if not texts_to_send and files and not sent:
        sent = await client.send_message (chat, file = files , message = '')
        txt = ''
    
    if not sent and poll:
        txt=""
        sent = await send_poll(chat, poll=poll)

            
    if (sent):
        print (f"Finished sending to {chat}: {txt[:20]}...")
    else:
        print (f"NOT sent!")
    return sent


# In[10]:


async def verify_dest_chats(chatlist):
    verif_chatlist = []
    dg_groups = await all_dialog_groups()
    for chat in chatlist:
        if type(chat)==int:
            try:
                ent = await client.get_entity(chat)
                verif_chatlist.append((ent.id, None))
            except Exception as e:
                print (f"Chat {chat} not found...")
            
        elif type(chat)==str:
            if chat in dg_groups.keys():
                appendable = [(i, None) for i in dg_groups[chat]]
                verif_chatlist.extend(appendable)
                print(f"Will send message to {chat}: total {len(dg_groups[chat])}!!!")
            elif ':' in chat:
                chat, thread = chat.split(':')
                chat = int(chat)
                thread = int(thread)
                verif_chatlist.append((chat, thread))
            else:
                print (f"Unknown chat: {chat}")

    
    return verif_chatlist


# In[11]:


def get_chat_type(chat):    
    if isinstance(chat, types.User):
        if chat.username == 'replies':
            return 'discussion'
        if chat.bot:
            return 'bot'
        else:
            return 'personal'

    elif isinstance(chat, types.ChannelForbidden):
        print (f"This is a message in a Forbidden Channel!")
        return 'channel'
    elif isinstance(chat, types.Channel):
        if chat.megagroup:
#            print (f"This is a message in Chat-megagroup!")
            return 'group'
        if chat.gigagroup:
#            print (f"This is a message in Chat-gigagroup!")
            return 'group'
        return 'channel'
        
    elif isinstance(chat, types.Chat):
#        print (f"This is a message in Chat!")
        return 'group'
    else:
        print(type(chat))
        return 'unknown!'


# In[12]:


async def all_dialog_groups():
    cg = {'all_bots':[], 'all_users':[], 'all_channels':[], 'all_groups':[]}
    dialogs = await client.get_dialogs ()
    for dg in dialogs:
        t = get_chat_type(dg.entity)
        if t=="bot": cg['all_bots'].append(dg.id)
        elif t=="personal": cg['all_users'].append(dg.id)
        elif t=="channel": cg['all_channels'].append(dg.id)
        elif t=="group": cg['all_groups'].append(dg.id)
        else:
            print("Non-working chat type: ",dg.id, t)
    return cg


# In[13]:


def non_english_symbols_regex(string: str):
    pattern = re.compile(r'[^\x20-\x7E]')
    return pattern.findall(string)


# In[14]:


def estimate_token_count(string: str, encoding_name = "gpt2") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# In[15]:


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


# In[16]:


def translate_to_lang(texts, target_lang):
    response = translate_service.translations().list(q=texts,target=target_lang).execute()
    return [i['translatedText'] for i in response['translations']][0]


# In[17]:


def extract_audio(in_video_path):
    try:
        sep = '\\' if len(in_video_path.split('\\')) > len(in_video_path.split('/')) else '/'
        aud_path = os.path.join(*in_video_path.split(sep)[:-1], 'audio_'+'.'.join(in_video_path.split(sep)[-1].split('.')[:-1])+'.mp3')
        aud = AudioSegment.from_file(in_video_path, "mp4")
        aud.export(aud_path, format="mp3")
        rate = aud.frame_rate
        print (f"Extracted audio to: {aud_path}")
        return aud_path, rate
    except Exception as e:
        aud_file = None
        print (f"Could NOT extract audio: {str(e)}")
        return None, 0


# In[18]:


def merge_media(vide_file, audio_file, res_file, fps=25, ar=0):
    print (f"Merging video: {vide_file} + audio: {audio_file}")
    if (audio_file):
        aud_cmd = "-i", audio_file
    else:
        aud_cmd = ""
        
    if (ar):
        rate_cmd = "-ar", str(ar)
    else:
        rate_cmd = ""
            
    command = [
        "ffmpeg",
        "-i", vide_file,
        "-r", str(fps),

        *aud_cmd,
        "-c:v", "copy",
        "-c:a", "aac",
        *rate_cmd,
        "-strict", "experimental",
        "-y",
        res_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print (f"Merged to {res_file}")
        return res_file
    except subprocess.CalledProcessError as error:
        print("Error: FFmpeg command failed with return code", error.returncode)
        print("Error output:", error.stderr.decode(), file=sys.stderr)
        return vide_file
    


# In[19]:


async def process_with_openai2 (rules, text):
    msg_list = [{'role':'system','content': rules}, {'role':'user', 'content':text}]
    req_tokens = 0
    for i in msg_list:
        req_tokens +=3
        req_tokens += estimate_token_count(i['content'])

    max_tokens = min(max(MAX_REQUEST_LENGHT - CONTIGENCY - req_tokens, 200), MAX_REQUEST_LENGHT - CONTIGENCY)
    max_tokens = min (max_tokens, int(req_tokens * (1 + ANSWER_RATIO)))
    
    try:
        resp = await openai.ChatCompletion.acreate(
          model="gpt-3.5-turbo",
          max_tokens=max_tokens,
          messages=msg_list,
          n=1,
          stop="###",
          temperature=TEMPERATURE
        )
        token_used = int(resp["usage"]["total_tokens"] )
        resp = str(resp.choices[0].message['content']).strip()
    except Exception as e:
        resp = (f"OpenAI error: {str(e)}")
        token_used = 0
    return resp, token_used



# In[20]:


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
        print (f"Could NOT rotate video: {str(e)}")
        return in_file


# In[21]:


def get_vid_params(vidfile):
    cap = cv2.VideoCapture(vidfile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = (cap.get(5))
    cap.release()
    return width, height, fps


# In[ ]:





# In[22]:


def make_work_wm(wmfile, frame_height, frame_width, scale=0.2, of=10, loc='br', wd='watermarks'):
    try:
        print (f"Making work_wm from {wmfile} with W {frame_width}, H {frame_height}")
        watermark = cv2.imread(wmfile, cv2.IMREAD_UNCHANGED)
        wm_height, wm_width, _ = watermark.shape
        aspect_ratio = wm_width / wm_height
        minsidelen = min (frame_height, frame_width)
        height, width, wm_chans = watermark.shape
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
        else:
            if loc[0] == 'b':
                y_offset = frame_height - new_height - of
            if loc[1] == 'r':
                x_offset = frame_width - new_width - of
        result = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = watermark
        resfile = os.path.join(WORKDIR, wd, f'wm_{str(random.random())[2:6]}.png')
        cv2.imwrite(resfile, result)
        print (f"Made watermark: {resfile}")
        return resfile
    except Exception as e:
        print (f"Could NOT make watermark, using as is: {str(e)}")
        return wmfile



# In[23]:


def flatten_video_fps(vidfile):
    sep = '\\' if len(vidfile.split('\\')) > len(vidfile.split('/')) else '/'
    retfile = os.path.join(*vidfile.split(sep)[:-1], 'flatfps_'+vidfile.split(sep)[-1])
    
    video = cv2.VideoCapture(vidfile)
    fps = video.get(cv2.CAP_PROP_FPS)
    fps = str(int(fps))
    return vidfile, fps
    
    print (f"Flattening video FPS: {vidfile}, will write to {retfile}")
    command = [
        "ffmpeg",
        "-i", vidfile,
        "-vf", "setpts=1/(25*TB)",
        "-r", fps,
        retfile
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print (f"Flattened video FPS: {vidfile} to {retfile}")
        return retfile, fps
    except subprocess.CalledProcessError as error:
        print("Error: FFmpeg command failed with return code", error.returncode)
        print("Error output:", error.stderr.decode(), file=sys.stderr)
        return vidfile, fps


# In[24]:


def add_wm_to_vid(vidfile, wmfile, fps=0):
    sep = '\\' if len(vidfile.split('\\')) > len(vidfile.split('/')) else '/'
    retfile = os.path.join(*vidfile.split(sep)[:-1], 'wm_'+vidfile.split(sep)[-1])
    if (fps):
        fps = "-r", fps
    else:
        fps = ""
        
    
    print (f"Adding WM to video: {wmfile} to {vidfile}, will write to {retfile}")
    command = [
        "ffmpeg",
        "-i", vidfile,
        *fps,
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


# In[25]:


def make_some_noise4(in_file, noise, hzrate):
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
#    out_file = out_sound.set_frame_rate(hzrate)  #

    out_sound.export(out_file, format="mp3")
    return out_file


# In[26]:


async def process_video(msg_vid, vid_rules, wd):
    if min(([not i for i in vid_rules.values()])):
        print (f"NO changes to video to be done, returning as is.")
        return msg_vid
    
    try:
        vidfile = os.path.join(WORKDIR, wd, f'invideo_{str(random.random())[2:5]}.mp4')
        print (f"Downloading video to {vidfile}!")
        await client.download_media(msg_vid, vidfile)
        w, h, fps = get_vid_params(vidfile)
        sep = '\\' if len(vidfile.split('\\')) > len(vidfile.split('/')) else '/'
        out_file = os.path.join(*vidfile.split(sep)[:-1], 'out_'+vidfile.split(sep)[-1])
    except Exception as e:
        print (f"Could NOT get initial file: {str(e)}")
        return msg_vid
    
    aud_file, hzrate = extract_audio(vidfile)
    vidfile, fps = flatten_video_fps(vidfile)
    
    if vid_rules.get('angle'):
        vidfile = rotate_video(vidfile, vid_rules['angle'])
    if (vid_rules.get('watermark')):
        wmloc = vid_rules.get('wm_loc')
        if not wmloc or wmloc not in ['br','bl','tr','tl', 'c']:
            wmloc = 'br'
            print (f"wm_loc incorrect, using default: bottom-right ('br')")
        wmfile = make_work_wm (vid_rules['watermark'], h, w, scale=0.2, of=10, loc=wmloc, wd=wd)
        vidfile = add_wm_to_vid(vidfile, wmfile)
        
    if (vid_rules['noise'] and aud_file):
        aud_file = make_some_noise4 (aud_file, vid_rules['noise'], hzrate)
    outpath = os.path.join(*vidfile.split(sep)[:-1], 'final_'+vidfile.split(sep)[-1])    
    video = merge_media(vidfile, aud_file, outpath, fps, hzrate)
    return video


# In[27]:


async def process_photo(msg_photo, img_rules, wd):
    
    if min(([not i for i in img_rules.values()])):
        print (f"NO changes to image to be done, returning as is.")
        return msg_photo

    
    photo_path = os.path.join(WORKDIR, wd, f'input_{str(random.random())[2:5]}.jpg')
    try:
        await client.download_media(msg_photo, photo_path)
    except Exception as e:
        print (f"Could NOT download image, skipping it...")
        return msg_photo

    
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
        wmfile = make_work_wm (img_rules['watermark'], height, width, scale=0.2, of=10, loc=wmloc, wd=wd)
        
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


# In[28]:


async def check_if_message_still_exists(msg, rules):
    fd = rules.get('fw_deleted') 
    if fd == 'YES':
        print ("Don't care if deleted - fw_deleted True")
        return True
    
    exists = False
    try:
        msglist = await client.get_messages(entity=msg.chat_id, ids=[msg.id])
        if msglist[0]:
            exists = True
    except:
        pass    

    if not exists and fd == 'ONLY_DELETED':
        print ("DELETED ONLY - will send it!")
        return True
        
    if exists and fd != 'ONLY_DELETED':
        return True
        
    return False


# In[29]:


async def process_album(update, rules):
    global busy
    
    channels = read_channels()
    
    files = []
    new_text = ""
    wd = f"{update.messages[0].chat_id}_{update.messages[0].id}"
    if wd not in os.listdir(WORKDIR):
        os.mkdir(os.path.join(WORKDIR, wd))
    
    
    msg_exists = None    
    while busy or msg_exists is None:
        msg_exists = await check_if_message_still_exists(update.messages[0], rules)
        if not msg_exists:
            print (f"Album in {update.messages[0].chat_id} # {update.messages[0].id} was deleted, will NOT continue")
            return
        await asyncio.sleep(3)
    busy = True

    

    destination_chats = [_['to'] for _ in channels if _['from']==update.chat_id][0]
    destination_chats = await verify_dest_chats(destination_chats)

    for message in update.messages:
        if message.photo:
            new_photo = await process_photo(message.photo, rules['image'], wd)
            files.append(new_photo)
        elif message.video:
            fw_file = await process_video(message.video, rules['video'], wd)
            files.append(fw_file)
        else:
            print (f"Some other message type: {message.to_dict()}")

        if message.text.strip():
            add_text = await get_processed_text (message.text, rules['text'])
            new_text += '\n'+add_text
                
    for (chat, thread) in destination_chats:
        sent = await send_msg(chat, text=new_text, files=files, thread = thread)
#        print ("Album sent:", sent)
        if rules.get("emote") and sent:
            emolist = await get_emotion_response(add_text)
            if (emolist):
                await react_to_message(chat, sent.id, emolist, thread=thread)
            else:
                print (f"Got NO emotions to use")


    busy = False     
    if (CLEANUP):
        shutil.rmtree(os.path.join(WORKDIR, wd), ignore_errors=False)


# In[30]:


def sched_vs_time(sched, ddt):
    for scheline in sched:
        if ddt.isoweekday() in scheline['days']:
            for rr in scheline['times']:
                if datetime.strptime(rr[0],"%H:%M").time() < ddt.time() < datetime.strptime(rr[1],"%H:%M").time():
                    return True
    return False


# In[31]:


def check_schedule(sched):
    ddt = datetime.now()
    to_process = False
    proc_time = ddt
    if sched:
        to_process = sched_vs_time(sched, ddt)
    else:
        to_process = True
        
    if not to_process:
        print ("Out of schedule!")
        for i in range (7*24*60):
            next_time = ddt+timedelta(minutes=i)
            proc_later = sched_vs_time(sched, next_time)
            if (proc_later):
                proc_time = next_time
                break
        
    return to_process, proc_time.replace(microsecond=0)


# In[32]:


async def album_callback(update: UpdateNewMessage):
    global processing_queue
    global upd
    
    channels = read_channels()
    upd = update.messages
    if update.chat_id in [_['from'] for _ in channels]:
        rules = [_['rules'] for _ in channels if _['from']==update.chat_id][0]
        receive_sched = [_.get('receive_shcedule') for _ in channels if _['from']==update.chat_id][0]
        receive, _ = check_schedule(receive_sched)
        if (receive):
            send_sched = [_.get('send_shcedule') for _ in channels if _['from']==update.chat_id][0]
            send_now, send_time = check_schedule(send_sched)
            fnc = process_album (update, rules)
            processing_queue.append((fnc, send_time))
            print (f"Added Album to processing (curr {len(processing_queue)}) to time {send_time}")
        else:
            print (F"Skipping Album in {update.message.chat_id} due to NON-receive shcedule!")


# In[33]:


async def processor():
    print (f"Processor started!")
    global processing_queue
    while True:
        returnables = []
        q = processing_queue.copy()
        for proc_params in q:
            proc, exec_time = proc_params
            if (exec_time <= datetime.now()):
                print (f"\n=========== Started processing! ============")
                await proc
                print (f"=========== FINISHED processing! ============\n")
                await asyncio.sleep(1)

            else:
                returnables.append((proc_params))
        for proc in q:
            processing_queue.remove(proc)
#        for pp in returnables:
#            print(f"Returned back to queue: {pp} ")
        if (returnables):
            processing_queue.extend(returnables)
#            print(f"Returned back to queue: {returnables} ")

        await asyncio.sleep(1)


# In[34]:


# Define the callback function that will be called when a new message arrives
async def message_callback(update: UpdateNewMessage):
    global processing_queue
    global upd
    upd = update
    
    channels = read_channels()
    message = update.message
#    print (message.to_dict())
    # Forward the message to another chat
    if update.message.chat_id in [_['from'] for _ in channels]:
        if message.grouped_id:
            return
        destination_chats = [_['to'] for _ in channels if _['from']==message.chat_id][0]
        destination_chats = await verify_dest_chats(destination_chats)

        rules = [_['rules'] for _ in channels if _['from']==message.chat_id][0]
        receive_sched = [_.get('receive_shcedule') for _ in channels if _['from']==update.chat_id][0]
        receive, _ = check_schedule(receive_sched)
        if (receive):
            for (chat, thread) in destination_chats:
                send_sched = [_.get('send_shcedule') for _ in channels if _['from']==update.chat_id][0]
                send_now, send_time = check_schedule(send_sched)
                fnc = forward_messages(chat, message, rules, thread)
                print (f"Added message to processing (curr {len(processing_queue)}) to time {send_time}")
                processing_queue.append((fnc, send_time))
        else:
            print (F"Skipping message in {update.message.chat_id} due to NON-receive shcedule!")


# In[35]:


async def forward_messages(destination_chat, message, rules, thread=None):  
    global busy
    global msg

    msg = message
    poll = None
    msg_exists = None    
    while busy or msg_exists is None:
        msg_exists = await check_if_message_still_exists(message, rules)
        if not msg_exists:
            print (f"Message in {message.chat_id} # {message.id} was deleted, will NOT continue")
            return
        await asyncio.sleep(3)
    busy = True
    
    if (message.text.strip()):
        new_text = await get_processed_text (message.text, rules['text'])
        if "Got error from OpenAI" in new_text:
            sent = await send_msg(ADMIN_ID, text=new_text)
            new_text = message.text.strip()
    else:
        print (f"text will be blank")
        new_text = ""
    
    fw_files = []
    wd = f"{message.chat_id}_{message.id}"
    if wd not in os.listdir(WORKDIR):
        os.mkdir(os.path.join(WORKDIR, wd))
    
    if (message.photo):
        fw_file = await process_photo(message.photo, rules['image'], wd)
        fw_files.append(fw_file)
    elif (message.video):
        fw_file = await process_video(message.video, rules['video'], wd)
        fw_files.append(fw_file)
    elif message.poll and rules['polls']:
        poll = message.poll
        
    sent_msg = await send_msg(destination_chat, files = fw_files, text=new_text, poll=poll,thread=thread)
    
    if rules.get("emote") and sent_msg and not poll:
        emolist = await get_emotion_response(new_text)
        if (emolist):
            await react_to_message(destination_chat, sent_msg.id, emolist, thread=thread)
        else:
            print (f"Got NO emotions to use")
        
    busy = False
    if (CLEANUP):
        shutil.rmtree(os.path.join(WORKDIR, wd), ignore_errors=False)


# In[36]:


async def get_processed_text(text, rules):
    if rules =="":
        print (f"Forwarding text as is: {text[:20]}")
        return text
    if text =="":
        print (f"NO text...")
        return ""
        
    ret_text = text
    if rules['prompt']:
        print (f"Rewriting text with TXT-rule: '{rules['prompt']}' :{text[:20]}...")    
        text_list = text_to_chunks(text)

        ret_text = ""
        total_tokens = 0
        for text in text_list:
            new_text, tokens = await process_with_openai2(text, rules['prompt'])
            ret_text+= f"\n{new_text}"
            total_tokens += tokens
        print (f"Rewrote using {total_tokens} tokens to: {ret_text[:20]}...")
        
    if rules['translate']:
        ret_text = translate_to_lang(ret_text,rules['translate'])
        print (f"Translated to {rules['translate']}: {ret_text[:20]}...")

    
    return ret_text
    


# In[37]:


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


# In[38]:


async def main():
    global me

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
        await client.sign_in(PHONE,password=password)


    dialogs = await client.get_dialogs ()
    all_dialogs = ""
    for dialog in dialogs:
        all_dialogs += f"\n{dialog.id} \t {dialog.title}"
    with open(f"{PHONE}_dialogs.txt", 'w', encoding='utf-8') as f:
        f.write(all_dialogs)
        
    if not me:
        me = await client.get_me()

    
    # Register the callback function
    client.add_event_handler(album_callback, events.Album)
    client.add_event_handler(message_callback, events.NewMessage)
    

    print (f"Bot up and running")
    # Start polling
    await client.run_until_disconnected()
    


# In[39]:


async def react_to_message(peer, msg_id, emolist, thread=None):
    global me
    if not me:
        me = await client.get_me()
    try:
        if not me.premium:
            emolist=emolist[:1]
        peer = await client.get_entity(peer)
        rr = await client(functions.messages.SendReactionRequest(peer=peer, msg_id=msg_id, 
                    add_to_recent=True, big=True,
#                        reply_to_msg_id = thread,
                    reaction=[(types.ReactionEmoji(emoticon=emote)) for emote in emolist]))
        print (f"Emoted post with {emolist}")
    except Exception as e:
        print (f"Could NOT react in {peer}: {str(e)}")


# In[40]:


if WORKDIR not in os.listdir():
    os.mkdir(WORKDIR)
    
global processing_queue
global busy
global me
processing_queue = []
busy = False
me = None


# In[41]:


openai.api_key = openai_key
client = TelegramClient(PHONE, API_ID, API_HASH)
translate_service = get_tanslate_service()
channels = read_channels()

workers = [main(), processor()]
async def runme (workers):
    await asyncio.gather(*workers)

if __name__ == '__main__':
    asyncio.run(runme(workers))
# In[ ]:


