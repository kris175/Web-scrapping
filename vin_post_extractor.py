from bs4 import BeautifulSoup
import pandas as pd

valid_posts = []
last_page_num = 798

for i in range(1,last_page_num+1):

    with open('TMC_pages/tmc_' + str(i), 'r', encoding='utf8') as html_file:
        content = html_file.read()
    soup = BeautifulSoup(content, 'lxml')

    cards =  soup.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['bbWrapper'])

    for text in cards:
        post = ''.join(list(text.find_all(text=True, recursive = False))).replace('\n'," ").lower()
        if " vin " in post: valid_posts.append(post)

valid_posts_df = pd.DataFrame({'text': valid_posts})
valid_posts_df.to_csv("tmc_vin_posts.csv")
