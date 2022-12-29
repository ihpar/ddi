from flask import Flask, render_template, request, jsonify
import pickle
import re
import string

clf, vectorizer = None, None
with open("tree.pkl", "rb") as f:
    clf = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def clean_text(text):
    stop_words = ["a", "acaba", "altı", "ama", "ancak", "artık", "asla", "aslında", "az", "b", "bana", "bazen", "bazı", "bazıları", "bazısı", "belki", "ben", "beni", "benim", "beş", "bile", "bir", "birçoğu", "birçok", "birçokları", "biri", "birisi", "birkaç", "birkaçı", "birşey", "birşeyi", "biz", "bize", "bizi", "bizim", "böyle", "böylece", "bu", "buna", "bunda", "bundan", "bunu", "bunun", "burada", "bütün", "c", "ç", "çoğu", "çoğuna", "çoğunu", "çok", "çünkü", "d", "da", "daha", "de", "değil", "demek", "diğer", "diğeri", "diğerleri", "diye", "dokuz", "dolayı", "dört", "e", "elbette", "en", "f", "fakat", "falan", "felan", "filan", "g", "gene", "gibi", "ğ", "h", "hâlâ", "hangi", "hangisi", "hani", "hatta", "hem", "henüz", "hep", "hepsi", "hepsine", "hepsini", "her", "her biri", "herkes", "herkese", "herkesi", "hiç", "hiç kimse", "hiçbiri", "hiçbirine", "hiçbirini", "ı", "i", "için", "içinde", "iki", "ile", "ise",
                  "işte", "j", "k", "kaç", "kadar", "kendi", "kendine", "kendini", "ki", "kim", "kime", "kimi", "kimin", "kimisi", "l", "m", "madem", "mı", "mı", "mi", "mu", "mu", "mü", "mü", "n", "nasıl", "ne", "ne kadar", "ne zaman", "neden", "nedir", "nerde", "nerede", "nereden", "nereye", "nesi", "neyse", "niçin", "niye", "o", "on", "ona", "ondan", "onlar", "onlara", "onlardan", "onların", "onların", "onu", "onun", "orada", "oysa", "oysaki", "ö", "öbürü", "ön", "önce", "ötürü", "öyle", "p", "r", "rağmen", "s", "sana", "sekiz", "sen", "senden", "seni", "senin", "siz", "sizden", "size", "sizi", "sizin", "son", "sonra", "ş", "şayet", "şey", "şeyden", "şeye", "şeyi", "şeyler", "şimdi", "şöyle", "şu", "şuna", "şunda", "şundan", "şunlar", "şunu", "şunun", "t", "tabi", "tamam", "tüm", "tümü", "u", "ü", "üç", "üzere", "v", "var", "ve", "veya", "veyahut", "y", "ya", "ya da", "yani", "yedi", "yerine", "yine", "yoksa", "z", "zaten"]

    text = " ".join(text.split())
    text = text.lower()
    text = text.replace("\\n", " ")
    text = re.sub("[0-9]+", "", text)
    ext = re.sub("%|(|)|-", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = [t for t in text.split() if t not in stop_words]
    return " ".join(text)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/classify-texts", methods=["POST"])
def yep():
    data = request.get_json()
    comments = data["texts"]
    comments_clean = [clean_text(c) for c in comments]
    vectors = vectorizer.transform(comments_clean)
    predictions = clf.predict(vectors).tolist()
    response = {
        "texts": comments,
        "classes": predictions
    }
    return jsonify(response)


app.run(host="0.0.0.0", port=8080)
