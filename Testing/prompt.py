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
    

if __name__ == '__main__':
    from sky_model import generate_response
    import warnings
    from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq
    import torch
    
    warnings.filterwarnings("ignore")

    model_name = '../Training/Saved_Models/Sky/fine-tuned-bert-sentiment_2024_09_28_0'
    tokenizer, model, data_collator, device = load_model_and_tokenizer(model_name)
    
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    flag = True
    conversationList = []
    previous_reponse = ""
    while flag:
        conversation = ""
        recordedText = "agent_1"+input("Ask me anything : ")
        
        if len(conversationList) > 2:
            conversationList = conversationList[-2:]
        # conversation = ''.join(conversationList)+" agent_1:"+recordedText
        print(recordedText)
        print()
        response = generate_response(recordedText, tokenizer, device, model)
        # similarity_score = calculate_response_similarity(previous_reponse,response)
        # if similarity_score > 0.5:
        #     response = generate_response("agent_1:"+recordedText, tokenizer, device, model)
        # previous_reponse = response.split(",<Emotion>:")[0].split("<Content>:")[1]
        # conversation = conversation+" agent_2:"+response.split(",<Emotion>:")[0].split("<Content>:")[1]
        # conversationList.append("agent_1:"+recordedText+" agent_2:"+response.split(",<Emotion>:")[0].split("<Content>:")[1])
        # print(response.split(",<Emotion>:")[0].split("<Content>:")[1])
        print(response)
        # print(conversation)
        # print(conversationList)
# Optionally, release GPU memory
    torch.cuda.empty_cache()
