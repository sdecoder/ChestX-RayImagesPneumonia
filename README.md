<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="resources/banner.png" alt="Logo">
  </a>

<h3 align="center">Chest X-Ray Images (Pneumonia)</h3>
  <!--
  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>-->
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About This Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

<!--
[![Product Name Screen Shot][product-screenshot]](https://example.com)
-->

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (
Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five
years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part
of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing
all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being
cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a
third expert.

Source:

* Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep
  Learning: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
* https://www.kaggle.com/c/cassava-leaf-disease-classification

### Prerequisites

Hardware platform:

* NVIDIA Tesla M40 24GB for inference usage
* AMD 6900XT 16GB for Pytorch ROCm[HIP] for training usage

Software platform:

* NVIDIA SDK: TensorRT/CUDA/CuDNN
* Python environment: Anaconda
* Deep learning framework: Pytorch
* OS: Arch Linux

<p align="right">(<a href="#top">back to top</a>)</p>

## Workflow

### Model training

The model[tf_efficientnet_b4_ns] structure is list as following:

This model is trained using Pytorch ROCm.
The training result can reach 99% precision.
After training, the weight is saved to the .pt file.

### Export to ONNX file

Reload the .pt file to construct the original Pytorch model.
Export the pytorch model to the .onnx format using following code snippet.

```python
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# Input to the model
input_data = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
output_data = torch_model(input)
output_file_name = "target.onnx"
# Export the model
torch.onnx.export(torch_model,  # model being run
                  input_data,  # model input (or a tuple for multiple inputs)
                  output_file_name,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})

```

Once the exporting is done, we can load the .onnx file for validation purpose.
The following snippet is used to load the .onnx file and use it as the inference engine.

```python
import onnx

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")


def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

```

### Export to TensorRT engine file

Once the conversion and validation for the .onnx file is done,
we can convert it to the TensorRT engine at this point.
Once the TensorRT engine is generated, we run the test again to make sure its precision keeps.

In our experiment setup, all INT8/FP16/TF32/FP32 configuration remains at 99% precision.
Note: If the selected precision is INT8, the calibrator dataset should be provided.

### Engine file size comparison

* classifier-sim.INT8.engine = 6.3M
* classifier-sim.FP16.engine = 3.2M
* classifier-sim.FP32.engine = 3.2M
* classifier-sim.TF32.engine = 3.2M

### Performance evaluation

The performance evaluation is done using a Tesla M40 card.
Corresponding specification can be found here:
https://www.techpowerup.com/gpu-specs/tesla-m40-24-gb.c3838

1. for the INT8 throughput:

```shell
trtexec --loadEngine=classifier-sim.INT8.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/25/2022-19:02:56] [I] === Performance summary ===
[07/25/2022-19:02:56] [I] Throughput: 2.54521e+06 qps
[07/25/2022-19:02:56] [I] Latency: min = 9.47876 ms, max = 51.4163 ms, mean = 16.5801 ms, median = 11.6239 ms, percentile(99%) = 47.6256 ms
[07/25/2022-19:02:56] [I] Enqueue Time: min = 0.00854492 ms, max = 0.0578613 ms, mean = 0.012563 ms, median = 0.00976562 ms, percentile(99%) = 0.0471191 ms
[07/25/2022-19:02:56] [I] H2D Latency: min = 3.12537 ms, max = 44.6831 ms, mean = 9.18409 ms, median = 4.26877 ms, percentile(99%) = 40.4702 ms
[07/25/2022-19:02:56] [I] GPU Compute Time: min = 5.99756 ms, max = 9.45909 ms, mean = 7.35231 ms, median = 7.1825 ms, percentile(99%) = 9.2168 ms
[07/25/2022-19:02:56] [I] D2H Latency: min = 0.00561523 ms, max = 0.0736084 ms, mean = 0.0437097 ms, median = 0.0429077 ms, percentile(99%) = 0.0670166 ms
[07/25/2022-19:02:56] [I] Total Host Walltime: 3.07697 s
[07/25/2022-19:02:56] [I] Total GPU Compute Time: 7.02881 s
[07/25/2022-19:02:56] [W] * GPU compute time is unstable, with coefficient of variance = 7.58229%.
[07/25/2022-19:02:56] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/25/2022-19:02:56] [I] Explanations of the performance metrics are printed in the verbose logs.
```

2. for the FP16 throughput:

```shell
trtexec --loadEngine=classifier-sim.FP16.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/25/2022-19:03:30] [I] Throughput: 2.5563e+06 qps
[07/25/2022-19:03:30] [I] Latency: min = 9.49252 ms, max = 52.554 ms, mean = 16.9202 ms, median = 12.3535 ms, percentile(99%) = 47.1478 ms
[07/25/2022-19:03:30] [I] Enqueue Time: min = 0.0170898 ms, max = 0.116455 ms, mean = 0.0237389 ms, median = 0.0185547 ms, percentile(99%) = 0.0854492 ms
[07/25/2022-19:03:30] [I] H2D Latency: min = 3.13013 ms, max = 45.7395 ms, mean = 9.50471 ms, median = 4.79257 ms, percentile(99%) = 40.3545 ms
[07/25/2022-19:03:30] [I] GPU Compute Time: min = 5.99194 ms, max = 9.51074 ms, mean = 7.37206 ms, median = 7.18329 ms, percentile(99%) = 9.13654 ms
[07/25/2022-19:03:30] [I] D2H Latency: min = 0.00585938 ms, max = 0.0742798 ms, mean = 0.0433958 ms, median = 0.0419922 ms, percentile(99%) = 0.0671387 ms
[07/25/2022-19:03:30] [I] Total Host Walltime: 3.05081 s
[07/25/2022-19:03:30] [I] Total GPU Compute Time: 7.0182 s
[07/25/2022-19:03:30] [W] * GPU compute time is unstable, with coefficient of variance = 7.65987%.
[07/25/2022-19:03:30] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/25/2022-19:03:30] [I] Explanations of the performance metrics are printed in the verbose logs.
```

3. for the TF32 throughput:

```shell
trtexec --loadEngine=classifier-sim.TF32.engine --batch=8192 --streams=8 --verbose --avgRuns=10
 [07/25/2022-19:03:57] [I] Throughput: 2.5548e+06 qps
[07/25/2022-19:03:57] [I] Latency: min = 9.12402 ms, max = 53.5422 ms, mean = 16.5924 ms, median = 11.104 ms, percentile(99%) = 46.9604 ms
[07/25/2022-19:03:57] [I] Enqueue Time: min = 0.00976562 ms, max = 0.0761719 ms, mean = 0.0148766 ms, median = 0.0108643 ms, percentile(99%) = 0.0613403 ms
[07/25/2022-19:03:57] [I] H2D Latency: min = 3.11755 ms, max = 46.7709 ms, mean = 9.2178 ms, median = 3.23969 ms, percentile(99%) = 39.5508 ms
[07/25/2022-19:03:57] [I] GPU Compute Time: min = 5.94873 ms, max = 12.0564 ms, mean = 7.33046 ms, median = 7.1571 ms, percentile(99%) = 9.20398 ms
[07/25/2022-19:03:57] [I] D2H Latency: min = 0.00537109 ms, max = 0.0720215 ms, mean = 0.0441205 ms, median = 0.0429688 ms, percentile(99%) = 0.0669861 ms
[07/25/2022-19:03:57] [I] Total Host Walltime: 3.0526 s
[07/25/2022-19:03:57] [I] Total GPU Compute Time: 6.97859 s
[07/25/2022-19:03:57] [W] * GPU compute time is unstable, with coefficient of variance = 7.52974%.
[07/25/2022-19:03:57] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/25/2022-19:03:57] [I] Explanations of the performance metrics are printed in the verbose logs.
[07/25/2022-19:03:57] [V] 
```

4. for the FP32 throughput:

```shell
trtexec --loadEngine=classifier-sim.FP32.engine --batch=8192 --streams=8 --verbose --avgRuns=10
 [07/25/2022-19:04:18] [I] Throughput: 2.54996e+06 qps
[07/25/2022-19:04:18] [I] Latency: min = 9.11011 ms, max = 56.7799 ms, mean = 16.7815 ms, median = 11.3374 ms, percentile(99%) = 47.455 ms
[07/25/2022-19:04:18] [I] Enqueue Time: min = 0.0098877 ms, max = 0.0678711 ms, mean = 0.0148348 ms, median = 0.0112305 ms, percentile(99%) = 0.0598145 ms
[07/25/2022-19:04:18] [I] H2D Latency: min = 3.12762 ms, max = 49.9872 ms, mean = 9.40668 ms, median = 4.24805 ms, percentile(99%) = 40.4771 ms
[07/25/2022-19:04:18] [I] GPU Compute Time: min = 5.95483 ms, max = 10.9875 ms, mean = 7.3315 ms, median = 7.17285 ms, percentile(99%) = 9.1684 ms
[07/25/2022-19:04:18] [I] D2H Latency: min = 0.00561523 ms, max = 0.0722656 ms, mean = 0.0433322 ms, median = 0.0424805 ms, percentile(99%) = 0.0668182 ms
[07/25/2022-19:04:18] [I] Total Host Walltime: 3.06161 s
[07/25/2022-19:04:18] [I] Total GPU Compute Time: 6.98692 s
[07/25/2022-19:04:18] [W] * GPU compute time is unstable, with coefficient of variance = 7.28868%.
[07/25/2022-19:04:18] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/25/2022-19:04:18] [I] Explanations of the performance metrics are printed in the verbose logs.
```

### Accuracy evaluation

```log
[trace] validate the tensorrt engine file using ../models/classifier-sim.INT8.engine
[trace] final run loss by TensorRT: 18.15721794217825

[trace] validate the tensorrt engine file using ../models/classifier-sim.FP16.engine
[trace] final run loss by TensorRT: 18.15721845626831

[trace] validate the tensorrt engine file using ../models/classifier-sim.TF32.engine
[trace] final run loss by TensorRT: 18.157218277454376

[trace] validate the tensorrt engine file using ../models/classifier-sim.FP32.engine
[trace] final run loss by TensorRT: 18.157218001782894

```

### Conclusion:

1. The file size of classifier-sim.INT8.engine is almost double as other engine files. This needs further investigation.
2. In term of accuracy, four engines are very close, which means quantization causes negligible accuracy loss.
3. The FP16 quantization has the best performance, which is slightly better than others.

<!-- BUILIT WITH
### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [![Next][Next.js]][TensorRT]
<p align="right">(<a href="#top">back to top</a>)</p>
 -->

<!-- GETTING STARTED
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
-->


<!-- ROADMAP -->

## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Explorer more possibility

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (
and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge

[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge

[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members

[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge

[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers

[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge

[issues-url]: https://github.com/othneildrew/Best-README-Template/issues

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge

[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/othneildrew

[product-screenshot]: images/screenshot.png

[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white

[Next-url]: https://nextjs.org/

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB

[React-url]: https://reactjs.org/

[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D

[Vue-url]: https://vuejs.org/

[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white

[Angular-url]: https://angular.io/

[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00

[Svelte-url]: https://svelte.dev/

[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white

[Laravel-url]: https://laravel.com

[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white

[Bootstrap-url]: https://getbootstrap.com

[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white

[JQuery-url]: https://jquery.com

[TensorRT]: https://developer.nvidia.com/tensorrt

<!-- data

-->
