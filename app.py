from crypt import methods
import email
from email import message
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

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

@app.route('/submit_message', methods=['POST'])
def submit_message():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']

if __name__ == '__main__':
	app.run(debug = True, host='0.0.0.0', port=5000)