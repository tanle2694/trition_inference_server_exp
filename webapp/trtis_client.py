import random
import numpy as np
from PIL import Image
import sys
from functools import partial
import os

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc

from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


FLAGS = None


def parse_model_grpc(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                   len(input_metadata.shape)))

    if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
        (input_config.format != mc.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " +
                        mc.ModelInput.Format.Name(input_config.format) +
                        ", expecting " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                        " or " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_metadata.name,
            output_metadata.name, c, h, w, input_config.format,
            input_metadata.datatype)


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))
    if len(model_metadata['outputs']) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata['outputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    input_config = model_config['input'][0]
    output_metadata = model_metadata['outputs'][0]

    max_batch_size = 0
    if 'max_batch_size' in model_config:
        max_batch_size = model_config['max_batch_size']

    if output_metadata['datatype'] != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata['name'] + "' output type is " +
                        output_metadata['datatype'])

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata['shape']:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims (not counting the batch dimension),
    # either CHW or HWC
    input_batch_dim = (max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata['shape']) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata['name'],
                   len(input_metadata['shape'])))

    if ((input_config['format'] != "FORMAT_NCHW") and
        (input_config['format'] != "FORMAT_NHWC")):
        raise Exception("unexpected input format " + input_config['format'] +
                        ", expecting FORMAT_NCHW or FORMAT_NHWC")

    if input_config['format'] == "FORMAT_NHWC":
        h = input_metadata['shape'][1 if input_batch_dim else 0]
        w = input_metadata['shape'][2 if input_batch_dim else 1]
        c = input_metadata['shape'][3 if input_batch_dim else 2]
    else:
        c = input_metadata['shape'][1 if input_batch_dim else 0]
        h = input_metadata['shape'][2 if input_batch_dim else 1]
        w = input_metadata['shape'][3 if input_batch_dim else 2]

    return (max_batch_size, input_metadata['name'], output_metadata['name'], c,
            h, w, input_config['format'], input_metadata['datatype'])


def preprocess(img, format, dtype, c, h, w, scaling, protocol):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if protocol == "grpc":
        if format == mc.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled
    else:
        if format == "FORMAT_NCHW":
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(results, output_name, batch_size, batching):
    """
    Post-process results to show classifications.
    """
    predict, score = [], []
    output_array = results.as_numpy(output_name)
    if len(output_array) != batch_size:
        raise Exception("expected {} results, got {}".format(
            batch_size, len(output_array)))

    # Include special handling for non-batching models
    for results in output_array:
        if not batching:
            results = [results]
        for result in results:
            result = result.decode("utf-8")
            if output_array.dtype.type == np.object_:
                cls = "".join(chr(x) for x in result).split(':')
            else:
                cls = result.split(':')
            print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))
            predict.append(cls[2])
            score += [{"index": cls[2], "val": int(float(cls[0]))}]
            print({"index": cls[2], "val": int(float(cls[0]))})
    return predict, score



def requestGenerator(batched_image_data, input_name, output_name, dtype, model_name, model_version, classes=1):

    # Set the input data
    inputs = []

    inputs.append(
        grpcclient.InferInput(input_name, batched_image_data.shape, dtype))
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = []

    outputs.append(
        grpcclient.InferRequestedOutput(output_name,
                                        class_count=classes))

    yield inputs, outputs, model_name, model_version


def connect_to_server(server_host, server_port, model_name, model_version):
    url = f"{server_host}:{server_port}"
    verbose = False
    # connection = {"text": f"Connected to server {server_host}:{server_port}, model: {model_name}, "
    #                       f"version: {model_version}", "success": True}
    connection = {"text": "", "success": False}
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=verbose)
        connection = {"text": f"Connected to server {server_host}:{server_port}, model: {model_name}, "
                              f"version: {model_version}", "success": True}
    except Exception as e:
        connection = {"text": f"Client creation failed {server_host}:{server_port}", "success": False}
        return connection
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version)
    except InferenceServerException as e:
        connection = {"text": f"Failed to retrieve the metadata: {server_host}:{server_port}, model: {model_name}, "
                              f"version: {model_version}", "success": False}
        return connection

    try:
        model_config = triton_client.get_model_config(
            model_name=model_name, model_version=model_version)
    except InferenceServerException as e:
        connection = {"text": f"Failed to retrieve the config: {server_host}:{server_port}, model: {model_name}, "
                             f"version: {model_version}", "success": False}
        return connection

    return connection

def get_prediction(image_filename, server_host='10.0.64.132', server_port=8001,
                   model_name="inception_graphdef", model_version=None):
    url = f"{server_host}:{server_port}"
    verbose = False
    model_version = str(model_version)
    # model_name = "inception_graphdef"
    # model_version = "1"
    # image_filename = "mug.jpg"
    scaling = "INCEPTION"
    protocol = "grpc"
    batch_size = 1
    # model_version
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=verbose)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)
    try:
        model_config = triton_client.get_model_config(
            model_name=model_name, model_version=model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)
    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model_grpc(
        model_metadata, model_config.config)

    filenames = []
    if os.path.isdir(image_filename):
        filenames = [
            os.path.join(image_filename, f)
            for f in os.listdir(image_filename)
            if os.path.isfile(os.path.join(image_filename, f))
        ]
    else:
        filenames = [
            image_filename,
        ]

    filenames.sort()
    image_data = []
    for filename in filenames:
        img = Image.open(filename)
        image_data.append(
            preprocess(img, format, dtype, c, h, w, scaling,
                       protocol.lower()))

    requests = []
    responses = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False
    user_data = UserData()
    async_requests = []
    sent_count = 0
    while not last_request:
        input_filenames = []
        repeated_image_data = []
        for idx in range(batch_size):
            input_filenames.append(filenames[image_idx])
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True
        if max_batch_size > 0:
            batched_image_data = np.stack(repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                    batched_image_data, input_name, output_name, dtype, model_name, model_version):
                sent_count += 1
                triton_client.async_infer(
                    model_name,
                    inputs,
                    partial(completion_callback, user_data),
                    request_id=str(sent_count),
                    model_version=model_version,
                    outputs=outputs)
        except InferenceServerException as e:
            print("inference failed: " + str(e))
            sys.exit(1)
    processed_count = 0
    while processed_count < sent_count:
        (results, error) = user_data._completed_requests.get()
        processed_count += 1
        if error is not None:
            print("inference failed: " + str(error))
            sys.exit(1)
        responses.append(results)

    for response in responses:
        this_id = response.get_response().id
        print("Request {}, batch size {}".format(this_id, batch_size))
        predict, score = postprocess(response, output_name, batch_size, max_batch_size > 0)
    print("PASS")
    return predict, score

def random_image(img_path='/workspace/web_server/static/images'):
  """
  Pull a random image out of the small end2end-demo dataset

  :param savePath: the path to save the file to. If None, file is not saved
  :return 0: file selected
  :return 1: label selelcted
  """
  random_dir = random.choice(os.listdir(img_path))
  random_file = random.choice(os.listdir(img_path + '/' + random_dir))

  return img_path + '/' + random_dir + '/' + random_file, random_dir, 'static/images' + '/' + random_dir + '/' + random_file

# get_prediction()