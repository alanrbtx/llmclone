from datasets import Dataset
import pandas as pd

def process_tg_data(
        file_path, 
        user,
        output_format="dataset"):
    """
    Prepare data for clon training
    This function works only for telegram chats
    """


    df = pd.read_json(file_path)

    messages = []
    current_user = ''
    for sample in df["chats"]["list"]:
        for row in sample["messages"]:
            if row["text"] != '':
                username = row['from']
                if username != user:
                    username = "User"
                if username == user:
                    username = "Clone"  
                message = f"{username}: {row['text']}"

                
                if message.startswith('User:'):
                    if current_user != 'User':
                        current_user = 'User'
                        messages.append(message)
                    else:
                        messages[-1] += '. ' + message[len('User: '):]
                else:
                    if current_user != 'Clone':
                        current_user = 'Clone'
                        messages.append(message)
                    else:
                        messages[-1] += '\n ' + message[len('Clone: '):]

    size = 5
    num_steps = len(messages)/5
    samples = ["\n".join(messages[i*size:(i+1)*size]) for i in range(round(num_steps))]

    df = pd.DataFrame({"prompt": samples})

    if output_format == "dataset":
        dataset = Dataset.from_pandas(df)
        return dataset
    if output_format == "dataframe":
        return df