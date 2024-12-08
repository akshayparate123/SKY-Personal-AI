def load_model_and_tokenizer(model_name):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return tokenizer, model, device


def calculate_response_similarity(previous_response,current_response):
    if previous_reponse == "":
        return 0
    previous = previous_response.split(" ")
    similarity_counter = 0
    for i in previous:
        if i in current_response:
            similarity_counter = similarity_counter+1
        else:
            pass
    print(similarity_counter/len(previous))
    return similarity_counter/len(previous)
    
def call_action(recordedText):
    print("Calling action")
    if "screenshot" in recordedText:
        img_path = action.screenshot()
        return "You can find the screenshot at location : {}".format(img_path),"screenshot"
    elif "read my screen" in recordedText:
        text = action.read_text_from_image(action.screenshot())
        return text,"ocr"
    elif "jobs" in recordedText:
        action.fetch_jobs("Data Science jobs")
        return "I have store the fetched jobs on your desktop. Do you want me to tailor your resume and update the tracker?","jobs"
    elif "canvas" in recordedText:
        messages = action.canvas(recordedText)
        return messages,"canvas"
    else:
        context = action.fetch_from_internet(recordedText)
        return context,"rag"
    return False

if __name__ == '__main__':
    from RealtimeSTT import AudioToTextRecorder
    from sky_model import generate_response
    import TTS
    import actions as action
    import warnings
    from transformers import BartForConditionalGeneration, BartTokenizer
    import torch
    warnings.filterwarnings("ignore")
    from datetime import datetime
    import time
    # Get the current date and time
    now = datetime.now()
    current_date_time = now.strftime("%Y_%m_%d")
    model_name = '../Training/Saved_Models/Sky/fine-tuned-bert-sentiment_2024_10_04_0'
    
    tokenizer, model, device = load_model_and_tokenizer(model_name)
    
    recorder = AudioToTextRecorder(spinner=False, model="large-v2", language="en", ensure_sentence_ends_with_period=True, 
                                   enable_realtime_transcription=True, silero_deactivity_detection = True)
    f = open("content.txt", "a")
    f.write("\n{} Model Used : {}".format(current_date_time,model_name))
    f.close()

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    flag = True
    print("Say something...")
    conversationList = []
    previous_reponse = ""
    while flag:
        conversation = ""
        recordedText = recorder.text()
        # logger.info("Recorded Text : {}".format(recordedText))
        print(recordedText)
        if recordedText == "Stop." or recordedText == "Stop!":
            # logger.info("Speaking Stopped")
            TTS.stop_speaker()
        elif "Sky" in recordedText:
            f = open("content.txt", "a")
            f.write("\n{} Recorded Text: {}".format(current_date_time,recordedText))
            if len(conversationList) > 2:
                conversationList = conversationList[-2:]
            conversation = ''.join(conversationList)+" agent_1:"+recordedText
            
            if "jobs" in recordedText:
                TTS.say("Sure mam")
                time.sleep(2)
                TTS.say("Please do not close the window.")
            else:
                TTS.say("Sure mam")
            recordedText = " ".join(recordedText.split(" ")[2:])
            action_response,action_ = call_action(recordedText)
            if not action_response:
                continue
            if action_ == "ocr":
                final_query = "<context>{}agent_1:{}".format(action_response,"what "+"".join(recordedText.split("what")[1]))
            elif action_ == "rag":
                final_query = "<context>{}agent_1:{}".format(action_response["documents"][0][0],recordedText)
                action.display_image("https://files.realpython.com/media/Screenshot_2023-09-03_at_2.52.02_PM.5681ade2fa81.png")
            elif action_ == "jobs":
                TTS.say(action_response)
                continue
            elif action_ == "screenshot":
                TTS.say(action_response)
                continue
            elif action_ == "canvas":
                len_m = len(action_response["subject"])
                if  len_m== 0:
                    TTS.say("There are no pending messages")
                else:
                    TTS.say("There are total {} pending message for you".format(len_m))
                    for i in range(0,len_m):
                        TTS.say("Message from {} on {}, with subject {}. The message says : {}".format(action_response["sender_name"][i],"fifth december at 11:30 PM",action_response["subject"][i],action_response["message"][i]))
                continue
            print(recordedText)
            # final_query = "<context>{}agent_1:{}".format(action_response["documents"][0][0],recordedText)
            response = generate_response(final_query, tokenizer, device, model)
            # logger.info("1st response generated by AI: {}".format(response))
            # similarity_score = calculate_response_similarity(previous_reponse,response)
            # logger.info("Similarity Score: {}".format(similarity_score))
            # if similarity_score > 0.5:
            #     # logger.info("Reponse rejected due to high similarity score")
            #     response = generate_response("agent_1:"+recordedText, tokenizer, device, model)
            #     # logger.info("New Response: {}".format(response))
            if "agent_2" in response:
                previous_reponse = response.split("agent_2:")[1]
            else:
                previous_reponse = response
            conversation = conversation+response
            # logger.info("QA with context: {}".format(conversation))
            conversationList.append("agent_1:"+recordedText+response)
            print(response)
            TTS.say(previous_reponse)
            try:
                f = open("content.txt", "a")
                f.write("\n{} Context(Action): {}".format(current_date_time,action_response))
                f.write("\n{} Model Response: {}".format(current_date_time,response))
                f.close()
            except UnicodeEncodeError as e:
                print(e)
        else:
            pass
    # Optionally, release GPU memory
    torch.cuda.empty_cache()
    