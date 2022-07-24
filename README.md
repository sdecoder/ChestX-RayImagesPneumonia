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

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

Source:
* Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
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
torch.onnx.export(torch_model,               # model being run
                  input_data,                         # model input (or a tuple for multiple inputs)
                  output_file_name,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

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
* efficientnet_b4_ns.FP16.engine = 70M
* efficientnet_b4_ns.FP32.engine = 71M
* efficientnet_b4_ns.INT8.engine = 72M
* efficientnet_b4_ns.TF32.engine = 71M

### Performance evaluation

The performance evaluation is done using a Tesla M40 card.
Corresponding specification can be found here:
https://www.techpowerup.com/gpu-specs/tesla-m40-24-gb.c3838



1. for the INT8 throughput:
```shell
trtexec --loadEngine=efficientnet_b4_ns.INT8.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/24/2022-22:56:49] [I] === Performance summary ===
[07/24/2022-22:56:49] [I] Throughput: 37538.6 qps
[07/24/2022-22:56:49] [I] Latency: min = 852.214 ms, max = 1882.35 ms, mean = 1671.06 ms, median = 1755.81 ms, percentile(99%) = 1882.35 ms
[07/24/2022-22:56:49] [I] Enqueue Time: min = 0.960327 ms, max = 446.976 ms, mean = 12.8272 ms, median = 1.01074 ms, percentile(99%) = 446.976 ms
[07/24/2022-22:56:49] [I] H2D Latency: min = 16.5605 ms, max = 115.081 ms, mean = 39.1974 ms, median = 17.144 ms, percentile(99%) = 115.081 ms
[07/24/2022-22:56:49] [I] GPU Compute Time: min = 835.549 ms, max = 1768.13 ms, mean = 1631.84 ms, median = 1734.03 ms, percentile(99%) = 1768.13 ms
[07/24/2022-22:56:49] [I] D2H Latency: min = 0.00561523 ms, max = 0.0610352 ms, mean = 0.0146191 ms, median = 0.0117188 ms, percentile(99%) = 0.0610352 ms
[07/24/2022-22:56:49] [I] Total Host Walltime: 17.2401 s
[07/24/2022-22:56:49] [I] Total GPU Compute Time: 128.916 s
[07/24/2022-22:56:49] [W] * GPU compute time is unstable, with coefficient of variance = 14.8214%.
[07/24/2022-22:56:49] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/24/2022-22:56:49] [I] Explanations of the performance metrics are printed in the verbose logs.


```
2. for the FP16 throughput:

```shell
trtexec --loadEngine=efficientnet_b4_ns.FP16.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/24/2022-22:58:54] [I] === Performance summary ===
[07/24/2022-22:58:54] [I] Throughput: 45453.9 qps
[07/24/2022-22:58:54] [I] Latency: min = 459.32 ms, max = 1587.09 ms, mean = 1353.2 ms, median = 1446.63 ms, percentile(99%) = 1587.09 ms
[07/24/2022-22:58:54] [I] Enqueue Time: min = 1.09277 ms, max = 680.736 ms, mean = 18.5081 ms, median = 1.17969 ms, percentile(99%) = 680.736 ms
[07/24/2022-22:58:54] [I] H2D Latency: min = 16.8507 ms, max = 115.957 ms, mean = 38.6342 ms, median = 17.2344 ms, percentile(99%) = 115.957 ms
[07/24/2022-22:58:54] [I] GPU Compute Time: min = 442.143 ms, max = 1471.2 ms, mean = 1314.55 ms, median = 1421.78 ms, percentile(99%) = 1471.2 ms
[07/24/2022-22:58:54] [I] D2H Latency: min = 0.00585938 ms, max = 0.0742188 ms, mean = 0.0206005 ms, median = 0.015625 ms, percentile(99%) = 0.0742188 ms
[07/24/2022-22:58:54] [I] Total Host Walltime: 14.2379 s
[07/24/2022-22:58:54] [I] Total GPU Compute Time: 103.849 s
[07/24/2022-22:58:54] [W] * GPU compute time is unstable, with coefficient of variance = 19.1136%.
[07/24/2022-22:58:54] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/24/2022-22:58:54] [I] Explanations of the performance metrics are printed in the verbose logs.

```

3. for the TF32 throughput:

```shell
trtexec --loadEngine=efficientnet_b4_ns.TF32.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/24/2022-23:00:20] [I] === Performance summary ===
[07/24/2022-23:00:20] [I] Throughput: 41712.7 qps
[07/24/2022-23:00:20] [I] Latency: min = 774.744 ms, max = 1690.64 ms, mean = 1507.59 ms, median = 1587.07 ms, percentile(99%) = 1690.64 ms
[07/24/2022-23:00:20] [I] Enqueue Time: min = 0.755859 ms, max = 598.609 ms, mean = 8.41466 ms, median = 0.806152 ms, percentile(99%) = 598.609 ms
[07/24/2022-23:00:20] [I] H2D Latency: min = 16.7282 ms, max = 115.398 ms, mean = 41.6897 ms, median = 17.2109 ms, percentile(99%) = 115.398 ms
[07/24/2022-23:00:20] [I] GPU Compute Time: min = 739.689 ms, max = 1585.34 ms, mean = 1465.88 ms, median = 1565.21 ms, percentile(99%) = 1585.34 ms
[07/24/2022-23:00:20] [I] D2H Latency: min = 0.00585938 ms, max = 0.0732422 ms, mean = 0.0203981 ms, median = 0.015625 ms, percentile(99%) = 0.0732422 ms
[07/24/2022-23:00:20] [I] Total Host Walltime: 15.5149 s
[07/24/2022-23:00:20] [I] Total GPU Compute Time: 115.804 s
[07/24/2022-23:00:20] [W] * GPU compute time is unstable, with coefficient of variance = 16.5552%.
[07/24/2022-23:00:20] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/24/2022-23:00:20] [I] Explanations of the performance metrics are printed in the verbose logs.
```

4. for the FP32 throughput:

```shell
trtexec --loadEngine=efficientnet_b4_ns.FP32.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/24/2022-23:01:35] [I] === Performance summary ===
[07/24/2022-23:01:35] [I] Throughput: 36973.1 qps
[07/24/2022-23:01:35] [I] Latency: min = 570.521 ms, max = 1896.41 ms, mean = 1649.98 ms, median = 1772.86 ms, percentile(99%) = 1896.41 ms
[07/24/2022-23:01:35] [I] Enqueue Time: min = 1.15918 ms, max = 727.728 ms, mean = 10.4853 ms, median = 1.21777 ms, percentile(99%) = 727.728 ms
[07/24/2022-23:01:35] [I] H2D Latency: min = 16.8677 ms, max = 114.881 ms, mean = 37.5984 ms, median = 17.2188 ms, percentile(99%) = 114.881 ms
[07/24/2022-23:01:35] [I] GPU Compute Time: min = 553.484 ms, max = 1781.69 ms, mean = 1612.36 ms, median = 1750.15 ms, percentile(99%) = 1781.69 ms
[07/24/2022-23:01:35] [I] D2H Latency: min = 0.00585938 ms, max = 0.0794678 ms, mean = 0.0206948 ms, median = 0.0175781 ms, percentile(99%) = 0.0794678 ms
[07/24/2022-23:01:35] [I] Total Host Walltime: 17.5038 s
[07/24/2022-23:01:35] [I] Total GPU Compute Time: 127.377 s
[07/24/2022-23:01:35] [W] * GPU compute time is unstable, with coefficient of variance = 19.1903%.
[07/24/2022-23:01:35] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/24/2022-23:01:35] [I] Explanations of the performance metrics are printed in the verbose logs.

```

### Accuracy evaluation

```log
[trace] validate the tensorrt engine file using ../models/efficientnet_b4_ns.INT8.engine
[trace] validation multi-class accuracy = 0.9062

[trace] validate the tensorrt engine file using ../models/efficientnet_b4_ns.FP16.engine
[trace] validation multi-class accuracy = 0.9031

[trace] validate the tensorrt engine file using ../models/efficientnet_b4_ns.TF32.engine
[trace] validation multi-class accuracy = 0.8844

[trace] validate the tensorrt engine file using ../models/efficientnet_b4_ns.FP32.engine
[trace] validation multi-class accuracy = 0.8812

```

### Conclusion:

1. The file size of TensorRT engines don't vary too much for different date type.
2. The FP16 has achieved best performance. The improvement can reach 22.937% (45453.9 qps vs 36973.1 qps)
3. The FP16 quantization has little impact to the accuracy of the inference result.

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


See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

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
