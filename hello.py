from flask import Flask, render_template, request, jsonify
from preprocessing import preprocessing

app = Flask(__name__)
if __name__ == '__main__':
    app.run(debug=True)
app.config.update(
	# DEBUG=True,
	TEMPLATES_AUTO_RELOAD=True
	)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def execute():
	chat = preprocessing()
	# msg = request.args.get('msg')
	msg = request.get_json()
	# return msg
	# kata = "apakah stok ada"
	res = chat.predict(inputtxt=msg['msg'])
	# print(msg['msg'])
	return jsonify(res)
	# return msg
