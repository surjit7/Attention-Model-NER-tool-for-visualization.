from flask import Flask, render_template, url_for, request, jsonify
import spacy
from spacy import displacy
from flaskext.markdown import Markdown
from transformers import TokenClassificationPipeline, AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
pipe = TokenClassificationPipeline(model=model, tokenizer=tokenizer)

app = Flask(__name__)
Markdown(app)
colors = {'PER': '#0D96BA', 'DEG':'#FE1520','LOC':'#FFDE08','ORG':'#08B3AB','JOB':'#16267D','SPL':'#FF9A98'}
bg_colors ={'PER': '#ffc0cb'}
options = {"ents": ['PER','DEG','LOC','ORG','JOB','SPL'], "colors": colors, "bg":bg_colors}
output=""
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

def get_entities_html(text, ner_result, title=None):
    """Visualize NER with the help of SpaCy"""
    ents = []
    for ent in ner_result:
        e = {}
        # add the start and end positions of the entity
        e["start"] = ent["start"]
        e["end"] = ent["end"]
        # add the score if you want in the label
        #e["label"] = f"{ent['entity_group']}-{ent['score']:.2f}"
        e["label"] = ent["entity_group"]
        if ents and -1 <= ent["start"] - ents[-1]["end"] <= 1 and ents[-1]["label"] == e["label"]:
            # if the current entity is shared with previous entity
            # simply extend the entity end position instead of adding a new one
            ents[-1]["end"] = e["end"]
            continue
        ents.append(e)
    # construct data required for displacy.render() method
    render_data = [
        {
            "text": text,
            "ents": ents,
            "title": title,
        }
    ]
    return spacy.displacy.render(render_data, style="ent", manual=True, options=options)


@app.route('/')
def index():
    return render_template("index1.html")


@app.route('/', methods=["POST"])
def process():
    if request.method == 'POST':
        text = request.form['rawtext']
        print(text)
        output1 = pipe(text, ignore_labels=["O"], aggregation_strategy="max")
        print(output1)
        html = get_entities_html(text, output1)
        html = html.replace("\n\n", "\n")
        result = HTML_WRAPPER.format(html)
    return render_template("index.html", text=text, output=result)


if __name__ == '__main__':
    app.run(debug=True)