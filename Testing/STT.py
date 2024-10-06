
def load_model_and_tokenizer(model_name):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return tokenizer, model, data_collator, device


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
    

def call_action(recorderText):
    print("Calling action")
    if "screenshot" in recordedText:
        img_path = action.screenshot()
        return "You can find the screenshot at location : {}".format(img_path)
    elif "read my screen" in recordedText:
        text = action.read_text_from_image(action.screenshot())
        return text
    return False

if __name__ == '__main__':
    from RealtimeSTT import AudioToTextRecorder
    from sky_model import generate_response
    import TTS
    import actions as action
    import warnings
    from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq
    import torch
    warnings.filterwarnings("ignore")
    from datetime import datetime
    # Get the current date and time
    now = datetime.now()
    current_date_time = now.strftime("%Y_%m_%d")
    model_name = '../Training/Saved_Models/Sky/fine-tuned-bert-sentiment_2024_10_04_0'
    
    tokenizer, model, data_collator, device = load_model_and_tokenizer(model_name)
    
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
            TTS.say("Sure Sir")
            action_response = call_action(recordedText)
            if not action_response:
                continue
            response = generate_response("<summary>"+action_response, tokenizer, device, model)

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
            f = open("content.txt", "a")
            f.write("\n{} Context(Action): {}".format(current_date_time,action_response))
            f.write("\n{} Model Response: {}".format(current_date_time,response))
            f.close()
        else:
            pass
    # Optionally, release GPU memory
    torch.cuda.empty_cache()
    