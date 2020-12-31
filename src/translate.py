import six
from google.cloud import translate_v2 as translate
import json_lines
import json

translate_client = translate.Client()

def translate_en(text):
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    result = translate_client.translate(text, target_language='en', format_='text')

    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return result["detectedSourceLanguage"], result["translatedText"]

with open('data/reviews_112.jl', 'rb') as f:
    with open('data/reviews_112_trans-en.jl', 'wb') as fw:
        for item in json_lines.reader(f):
            tmp = item
            src_lang, trans_en = translate_en(item['text'])
            tmp['trans_en'] = trans_en
            tmp['src_lang'] = src_lang
            fw.write(json.dumps(tmp, ensure_ascii=False).encode('utf8'))
            fw.write(b'\n')