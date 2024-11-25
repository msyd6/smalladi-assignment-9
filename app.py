from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
import os
from neural_networks import visualize
import traceback

app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        activation = request.json['activation']
        lr = float(request.json['lr'])
        hidden_dim = int(request.json['hidden_dim'])
        step_num = int(request.json['step_num'])

        # Run the experiment with the provided parameters
        visualize(activation, lr, hidden_dim, step_num)

        # Check if result gif is generated and return its path
        result_gif = "results/visualize.gif"
        
        return jsonify({
            "result_gif": result_gif if os.path.exists(result_gif) else None,
        })
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    response = make_response(send_from_directory('results', filename))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    app.run(debug=True, port=3000)

