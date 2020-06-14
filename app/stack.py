from flask import Flask, request, render_template
from joblib import load
import os
from clean import clean
port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)

model = load('app/xgbooster')
vocab = load('app/vocab')

from sklearn.feature_extraction.text import TfidfVectorizer
transformer = TfidfVectorizer(min_df=2, max_df=0.6, smooth_idf=True,
                              norm = 'l2', ngram_range=[1,2], max_features=125000,
                              decode_error="replace", vocabulary=vocab)
transformer.fit_transform(vocab)

@app.route('/', methods=["GET", "POST"])
def index():
    '''Change value to predicted gender'''
    blogpost = None
    value = 0
    if request.method == 'POST':
        blogpost = request.form.get("blogpost", None)
        
    def get_vector(blogpost):
    	vector = transformer.transform([' '.join(clean(blogpost))])
    	return vector
    if blogpost:
        vector = get_vector(blogpost)
        value = model.predict(vector)
        if int(value) == 0:
            value = 'Female'
        else:
            value = 'Male'
        return render_template('app.html', blogpost=blogpost, value=value)
    return render_template('app.html')

if __name__ == '__main__':
    app.run(debug=False, host = "0.0.0.0", port=port)
