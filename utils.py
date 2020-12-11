import pandas as pd

def extract_website(urls):
    return pd.Series([url.split('/')[2] if len(url.split('/'))>2 else None for url in urls])
