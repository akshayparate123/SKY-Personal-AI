def load_model_and_tokenizer(model_name):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return tokenizer, model, device

def call_action(recordedText):
    print("Calling action")
    if "screenshot" in recordedText:
        img_path = action.screenshot()
        return "You can find the screenshot at location : {}".format(img_path)
    elif "read my screen" in recordedText:
        text = action.read_text_from_image(action.screenshot())
        return text
    else:
        context = action.fetch_from_internet(recordedText)
        return context
    return False

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
    

if __name__ == '__main__':
    from sky_model import generate_response
    import actions as action
    import warnings
    from transformers import BartForConditionalGeneration, BartTokenizer
    import torch
    warnings.filterwarnings("ignore")
    from datetime import datetime
    # Get the current date and time
    now = datetime.now()
    current_date_time = now.strftime("%Y_%m_%d")
    model_name = '../Training/Saved_Models/Sky/fine-tuned-bert-sentiment_2024_10_04_0'
    
    tokenizer, model, device = load_model_and_tokenizer(model_name)
    
    f = open("content.txt", "a")
    f.write("\n{} Model Used : {}".format(current_date_time,model_name))
    f.close()

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    flag = True
    conversationList = []
    previous_reponse = ""
    while flag:
        # conversation = ""
        # recordedText = "agent_1"+input("Ask me anything : ")
        recordedText = input("Ask me anything : ")
        f = open("content.txt", "a")
        f.write("\n{} Recorded Text: {}".format(current_date_time,recordedText))
        action_response = call_action(recordedText)
        if not action_response:
            continue
        print(recordedText)
        final_query = "<context>{}agent_1:{}".format(action_response["documents"][0][0],recordedText)
        # response = generate_response("<summary>"+action_response, tokenizer, device, model)
        response = generate_response(final_query, tokenizer, device, model)
        print(response)
        try:
            f = open("content.txt", "a")
            f.write("\n{} Context(Action): {}".format(current_date_time,action_response))
            f.write("\n{} Model Response: {}".format(current_date_time,response))
            f.close()
        except UnicodeEncodeError as e:
            print(e)


# Optionally, release GPU memory
    torch.cuda.empty_cache()
