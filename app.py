from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='html/', static_folder='.', static_url_path='')

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/aboutus.html')
def about():
	return render_template('aboutus.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
	app.run(debug = True, host='0.0.0.0', port=5000)