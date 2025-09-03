from dotenv import load_dotenv
import os
import requests
from urllib.parse import quote_plus
from snapshot import poll_snapshot_status, download_snapshot


load_dotenv()

dataset_id = "gd_lvz8ah06191smkebj4"

def make_api_request(url, **kwargs):
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url,headers=headers,**kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None 
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def serp_search(query, engine="google"):
    if engine == "google":
        base_url = "https://www.google.com/search"
    else:
        raise ValueError("Unsupported search engine")

    url = "https://api.brightdata.com/request"

    payload ={
        "zone": "serp_api1",
        "url": f"{base_url}?q={quote_plus(query)}&brd_json=1",
        "format":"raw"
    }

    full_response = make_api_request(url, json=payload)

    if not full_response:
        return None

    extracted_data = {
        "knowledge": full_response.get("knowledge", {}),
        "organic": full_response.get("organic", []),
    }

    return extracted_data


def _reddit_snapshot(trigger_url,params,data,operation_name ="operation"):
    trigger_result = make_api_request(trigger_url,params=params,json=data)
    if not trigger_result:
        return None
    
    snapshot_id = trigger_result.get("snapshot_id")
    if not snapshot_id:
        print("No snapshot_id found in trigger response")
        return None

    if not poll_snapshot_status(snapshot_id):
        return None
    
    raw_data = download_snapshot(snapshot_id)
    return raw_data

def reddit_search_api(keyword,date="All time", sort_by="Hot",num_posts=20):
    trigger_url = "https://api.brightdata.com/datasets/v3/trigger"
    params ={
        "dataset_id": "gd_lvz8ah06191smkebj4",
        "include_errors": "true",
        "type": "discover_new",
        "discover_by": "keyword",
    }
    
    data = [
        {
        "keyword": keyword,
        "date": date,
        "sort_by": sort_by,
        "num_of_posts": num_posts
        } 
    ]

    raw_data = _reddit_snapshot(trigger_url,params,data,operation_name="reddit")

    if not raw_data:
        return None
    
    parsed_data =[]
    
    for post in raw_data:
        parsed_post ={
            "title": post.get("title"),
            "url": post.get("url"),
        }
        parsed_data.append(parsed_post)
    
    return {"parsed_data": parsed_data, "total_posts": len(parsed_data)}


def reddit_post_retrieval(urls, days_back=10,load_all_comments=False,comment_limit=""):
    if not urls:
        return None
    trigger_url = "https://api.brightdata.com/datasets/v3/trigger"

    params={
        "dataset_id": "gd_lvzdpsdlw09j6t702",
        "include_errors": "true",
    }

    data = [
        {
            "url": url,
            "days_back": days_back,
            "load_all_replies": load_all_comments,
            "comment_limit": comment_limit
        }
        for url in urls
    ]

    raw_data = _reddit_snapshot(trigger_url,params,data,operation_name="reddit_post_retrieval")

    if not raw_data:
        return None
    
    parsed_comments =[]
    for comment in raw_data:
        parsed_comment ={
            "comment_id":comment.get("comment_id"),
            "content": comment.get("comment"),
            "date": comment.get("date_posted"),
        }
        parsed_comments.append(parsed_comment)

    return {"parsed_comments": parsed_comments, "total_comments": len(parsed_comments)}