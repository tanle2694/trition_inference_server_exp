'''
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import pathlib
import logging
import os
# from threading import Timer

from flask import Flask, render_template, request
from trtis_client import get_prediction, random_image, connect_to_server

app = Flask(__name__)

name_arg = os.getenv('MODEL_SERVE_NAME', 'resnet_graphdef')
addr_arg = os.getenv('TRTSERVER_HOST', '10.0.64.132')
port_arg = os.getenv('TRTSERVER_PORT', '8001')
model_version = os.getenv('MODEL_VERSION', '2')


# handle requests to the server
@app.route("/", methods=['POST', 'GET'])
def main():
  connection = None
  args = {"name": name_arg, "addr": addr_arg, "port": port_arg, "version": str(model_version)}
  return render_template('index_connect.html', connection=connection, args=args)


@app.route("/connect", methods=['POST'])
def connect():
  data = request.form.to_dict(flat=False)
  print(data)
  server_host = data['addr'][0]
  server_port = data['port'][0]
  model_name = data['name'][0]
  model_version = str(data['version'][0])
  args = {"name": model_name, "addr": server_host, "port": server_port, "version": model_version}
  connection = connect_to_server(server_host, server_port, model_name, model_version)
  output = None
  if connection['success']:
      current_dir = str(pathlib.Path(__file__).parent.absolute())
      print(os.path.join(current_dir, 'static/images/'))
      file_name, truth, serving_path = random_image(os.path.join(current_dir, 'static/images/'))
      # get prediction from TensorFlow server
      pred, scores = get_prediction(file_name,
                                    server_host=addr_arg,
                                    server_port=int(port_arg),
                                    model_name=name_arg,
                                    model_version=int(model_version))
      output = {"truth": truth, "prediction": pred,
            "img_path": serving_path, "scores": scores}
  return render_template('index.html', connection=connection, args=args, output=output)


def remove_resource(path):
  """
  attempt to delete file from path. Used to clean up MNIST testing images

  :param path: the path of the file to delete
  """
  try:
    os.remove(path)
    print("removed " + path)
  except OSError:
    print("no file at " + path)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO,
                      format=('%(levelname)s|%(asctime)s'
                              '|%(pathname)s|%(lineno)d| %(message)s'),
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      )
  logging.getLogger().setLevel(logging.INFO)
  logging.info("Starting flask.")
  app.run(debug=True, host='0.0.0.0', port=5001)
