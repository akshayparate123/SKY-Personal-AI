import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import networkx as nx
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import logging
from datetime import datetime
# Get the current date and time
now = datetime.now()
current_date_time = now.strftime("%Y_%m_%d")

logging.basicConfig(
    level=logging.INFO,  # Set the minimum level of messages to log
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of log messages
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.FileHandler("../logs/{}_{}.log".format("CollectData",current_date_time)),  # Log messages to a file
        # logging.StreamHandler()  # Also print log messages to console
    ]
)
logger = logging.getLogger(__name__)



def get_google_search_links(query):
    results = search(query)
    return [link for link in results]
def get_bing_search_links(query):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    ]
    url = 'https://www.bing.com/search?q={}'.format(query.replace(" ","+"))
    headers = {'User-Agent': user_agents[0]}
    r = requests.get(url,headers=headers)
    soup = BeautifulSoup(r.content, 'html5lib')
    results = soup.find("div",{'id':'b_content'})
    h2 = results.find_all("h2")
    li = []
    for i in h2:
        try:
            filter = i.find("a")["href"]
            if "https://" in filter:
                li.append(filter)
        except Exception as e:
            pass
    return li

def index_of_true(priority_tags):
    key_true = []
    for tag in priority_tags.keys():
        if priority_tags[tag] == True:
            key_true.append(tag)
        else:
            pass
    return key_true

def find_nested_links(tag):
    links = []
    nested_tags = tag.find_all('a')
    if len(nested_tags) != 0:
        for i in nested_tags:
            try:
                links.append(str(i).split(" ")[1].split("href=")[1]) #Storing the links
            except IndexError as index_error:
                logger.info(index_error)
                return links
    return links

def adjust_tags_based_on_priority(priority_tags,tag_name,record_tags,index_value):
    #if less priority tags are already present and new high priority tag comes, we will make all low priority tag to false and high priority tag true
    #if less priority tags comes and high priority tag is True, We will keep all same and make low priority task true
    values = list(priority_tags.values())
    keys = list(priority_tags.keys())
    all_indexes_with_true = []
    for i in range(0,len(values)):
        if values[i]:
            all_indexes_with_true.append(i)
        else:
            pass
    if len(all_indexes_with_true) == 0:
        #If no tags are true, we will make the target tag true and return the same.
        priority_tags[tag_name] = not priority_tags[tag_name]
        record_tags[tag_name].append(index_value)
        return priority_tags,record_tags,""
    else:
        priority_type = "high" #high or low
        idx = keys.index(tag_name)
        if idx <= min(all_indexes_with_true):
            priority_type = "high"
            for k in keys:
                priority_tags[k] = False
            priority_tags[tag_name] = True
            record_tags[tag_name].append(index_value)
            return priority_tags,record_tags,"high"
        elif idx >= max(all_indexes_with_true):
            priority_type = "low"
            priority_tags[tag_name] = True
            record_tags[tag_name].append(index_value)
            return priority_tags,record_tags,"low"
        else:
            priority_type = "mid"
            # print('Index of mid tag',idx)
            for k in keys[idx+1:]:
                priority_tags[k] = False
            priority_tags[tag_name] = True
            record_tags[tag_name].append(index_value)
            return priority_tags,record_tags,"mid"
            #Make all the tags with less priority than current tag to false. And make the current tag true

def create_network_graph(indexing,sub_indexing,nxG,type,priority_tags,parent_node,record_tags):
    if type == "high":
        nxG.add_edge(parent_node,indexing)
    elif type == "low" or type == "mid":
        key_true = index_of_true(priority_tags)[:-1]
        # print("KGLA_structure --> index_of_true",key_true[-1])
        new_edge = record_tags[key_true[-1]][-1]
        nxG.add_edge(new_edge,indexing)
        for v in sub_indexing:
            nxG.add_edge(indexing, v)
        return nxG
    else:
        nxG.add_edge(parent_node,indexing)
    for v in sub_indexing:
            nxG.add_edge(indexing, v)
    return nxG
def clean_data(link):
    try:
        r = requests.get(link,timeout=(3, 5))
        logger.info("request : {}".format(r))
        if r.status_code == 403:
            return [],""
        soup = BeautifulSoup(r.content, 'html5lib')
        h1_tag = str(soup.find('h1'))
        parent_node = soup.find('title').text       
        for tag in soup(['nav', 'header', 'footer', 'script', 'style', 'aside']):
            tag.decompose()
        imp_tags = soup.find_all(['h1', 'h2', 'h3', 'h4','h5','strong', 'p', 'li'])
        imp_tags.insert(0,BeautifulSoup(h1_tag, 'html5lib').find("h1"))
        return imp_tags,parent_node
    except Exception as e:
        logger.info(e)
        return [],""

def adjust_pending_task(nxG,pendingTopics,text_index):
    see_also = list(nxG.adj[0].keys())[1:]
    for i in see_also:
        if text_index[i] == 'See also':
            topics = list(nxG.adj[i].keys())
            pendingTopics.extend([text_index[j].lower().split("\xa0–")[0] for j in topics if text_index[j].lower().split("\xa0–")[0] not in pendingTopics])
        else:
            pass
    return pendingTopics
def generate_heading(hl,r,nxG):
    temp = []
    if len(hl) == 0:
        # print(r)
        return r
    else:
        for h in hl:
            # print(r)
            # print(list(nxG.adj[h]))
            temp.append(generate_heading(list(nxG.adj[h])[1:],r+'->'+str(h),nxG))
        return temp
    

def priority_based_structure(imp_tags,parent_node):
    try:
        priority_tags = {"h1":False,"h2":False,"h3":False,"h4":False,"h5":False,"strong":False,"p":False,"li":False,"a":False}
        record_tags = {"h1":[],"h2":[],"h3":[],"h4":[],"h5":[],"strong":[],"li":[],"p":[],"a":[]}
        text_index = {}
        indexing = 0
        links = []
        nxG = nx.Graph()
        nxG.add_node(parent_node)
        for idx,tag in enumerate(imp_tags):
            key_true = index_of_true(priority_tags)
            if len(key_true) == 0 and tag.name in ["ul","li","ol"]: #If there is no heading, We will find the link and store them directly in a list for futher scraping
                pass
                # links.extend(find_nested_links(tag))
            else:
                text_index[indexing] = tag.text
                # nested_links = find_nested_links(tag)
                # links.extend(nested_links)
                # step_size = 0.001
                sub_indexing = []
                priority_tags,record_tags,type = adjust_tags_based_on_priority(priority_tags,tag.name,record_tags,indexing)
                nxG = create_network_graph(indexing,sub_indexing,nxG,type,priority_tags,parent_node,record_tags)
                indexing = indexing+1
        return nxG,text_index
    except Exception as e:
        logger.info(e)
        nxG = nx.Graph()
        text_index = {}
        return nxG,text_index


def start(link,store_data,pendingTopics,completedTopics,topicName):
    logger.info("{}. Link : {}".format(idx,link))
    imp_tags,parent_node = clean_data(link)
    if len(imp_tags) == 0:
        logger.info("{}. Skipping".format(idx))
        return store_data,pendingTopics,completedTopics
    logger.info("{}. Fetched imp tags : {}".format(idx,len(imp_tags)))
    nxG,text_index = priority_based_structure(imp_tags,parent_node)
    if len(list(text_index.keys())) == 0:
        logger.info("{}. Skipping".format(idx))
        return store_data,pendingTopics,completedTopics
    logger.info("{}. Created Network Graph : {}".format(idx,nxG))
    completedTopics.append(topicName)
    logger.info("{}. Total Completed Topics : {}".format(idx,len(completedTopics)))
    pendingTopics = adjust_pending_task(nxG,pendingTopics,text_index)
    logger.info("{}. Updating Pending Topics : {}".format(idx,len(pendingTopics)))
    list_headings = []
    root_node = list(nxG.adj[parent_node])
    logger.info("{}. Root Node : {}".format(idx,root_node))
    for r in root_node:
        # print(list(nxG.adj[r]))
        hl = list(nxG.adj[r])
        list_headings.extend(generate_heading(hl[1:],str(r),nxG))
    logger.info("{}. Total List Headings : {}".format(idx,len(list_headings)))
    store_data["Topic_Name"].append(topicName)
    store_data["URL"].append(link)
    store_data["Text_Index"].append(text_index)
    store_data["Network"].append(list_headings)
    store_data["All_Tags"].append(imp_tags)
    logger.info("{}. Data Stored".format(idx))
    return store_data,pendingTopics,completedTopics

completedTopics = []
pendingTopics = ["coin"]

store_data = {"Topic_Name":[],"URL":[],"All_Tags":[],"Text_Index":[],"Network":[]}
counter = 1484

for idx,pending in enumerate(pendingTopics):
    print(idx,")",pending)
    # print(f'\rProgress: {idx}/{len(pendingTopics)} Topic : {pending}', end='', flush=True)
    if idx % 10 == 0 and idx!=0:
        logger.info("{}. Storing the data in tsv file".format(idx))
        counter = counter+1
        pd.DataFrame.from_dict(store_data).to_csv('../Data/GraphAgent/{}.tsv'.format(counter),sep='\t', index=False)
        logger.info("{}. Emptying the dict".format(idx))
        store_data = {"Topic_Name":[],"URL":[],"All_Tags":[],"Text_Index":[],"Network":[]}
        logger.info(store_data)
    if len(completedTopics) == 0:
        logger.info("{}. First Loop".format(idx))
        startLink = "https://en.wikipedia.org/wiki/{}".format(pendingTopics[0].replace(" ","_"))
        store_data,pendingTopics,completedTopics = start(startLink,store_data,pendingTopics,completedTopics,pendingTopics[0])
    else:
        try:
            logger.info("Google Search")
            links = get_google_search_links(pending)
        except Exception as e:
            logger.info("Bing search")
            links = get_bing_search_links(pending)
        wiki_link = "https://en.wikipedia.org/wiki/{}".format(pending.replace(" ","_"))
        if wiki_link not in links:
            links.append(wiki_link)
        else:
            pass
        for idx2,link in enumerate(links):
            # print(f'\rProgress: {idx}.{idx2}/{len(pendingTopics)} Topic : {pending}', end='', flush=True)
            if (".gov" not in link) and ("linkedin.com" not in link) and ("reddit.com" not in link):
                try:
                    store_data,pendingTopics,completedTopics = start(link,store_data,pendingTopics,completedTopics,pending)
                except Exception as e:
                    logger.info(e)
                    break
            else:
                pass
        # break