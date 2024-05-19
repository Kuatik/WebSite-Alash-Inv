from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='html/')

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/about')
def about():
	return render_template('aboutus.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
	app.run(debug = True, host='0.0.0.0', port=5000)