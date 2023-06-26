# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search


# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GET_MODEL = "gpt-3.5-turbo"
openai.api_key = "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"

df = pd.read_csv(r"C:\Users\IsmailOKIEHOMAR\OneDrive - ABYS MEDICAL\Bureau\text_note_book_embedded.csv")
df = df.drop(['Unnamed: 0'], axis=1)


df['embedding'] = df['embedding'].apply(ast.literal_eval)


# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["content"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


# # examples
# strings, relatednesses = strings_ranked_by_relatedness("curling gold medal", df, top_n=5)
# for string, relatedness in zip(strings, relatednesses):
#     print(f"{relatedness=:.3f}")
#     display(string)



def num_tokens(text: str, model: str = GET_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below document on the Abys Medical Cysware 4H. If the answer cannot be found in the document, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nAbys medical document:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GET_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Surgiverse, Cysware, Cysart, Terms of use and Abys Medical."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


import os
import openai
import gradio
prompt = "hello, How can i help you ?"

def message_and_history(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = ask(inp)
    history.append((input, output))
    return history, history
block = gradio.Blocks(theme=gradio.themes.Monochrome())
with block:
    gradio.Markdown("""<h1><center>SurgiverseGPT</center></h1>
    """)
    chatbot = gradio.Chatbot()
    message = gradio.Textbox(placeholder=prompt)
    state = gradio.State()
    submit = gradio.Button("SEND")
    submit.click(message_and_history, 
                 inputs=[message, state], 
                 outputs=[chatbot, state])
block.launch(debug = True)