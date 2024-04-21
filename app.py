from flask import Flask, request, send_file, jsonify
import subprocess
import csv

app = Flask(__name__)

# Enable debug mode
app.config['DEBUG'] = True

# Define route for receiving data, updating CSV, running main.py, and returning TensorFlow Lite model
@app.route('/update_model', methods=['POST'])
def update_csv_and_generate_tflite():
    try:
        data = request.get_json()  # Assuming JSON data is sent
        
        # Write received data to heatstroke.csv
        with open('heatstroke.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())  # Assuming all items in the list have the same keys
            writer.writerows([item for item in data])

        # Run main.py script
        subprocess.run(['python3', 'main.py'])

        # Return generated TensorFlow Lite file
        return send_file('heatguard.tflite', as_attachment=True)
    except Exception as e:
        # Log the exception for debugging
        app.logger.error('An error occurred: %s', str(e))
        return jsonify({'error': str(e)}), 500  # Return error message with status code 500

if __name__ == '__main__':
    app.run(debug=True)  # Run the server in debug mode

