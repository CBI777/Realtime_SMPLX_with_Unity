# Realtime_SMPLX_with_Unity
<h2 align="center">
Realtime Single RGB - Pose / shape estimation with SMPL model in Unity
</h2>

------------
#About This Repository

Human pose and shape estimation from single RGB image was - and is - a topic that many researchers are intreseted in.
After the presentation of [SMPL](https://github.com/vchoutas/smplx) model, several deep learning models that does pose/shape estimation have been presented.
Models such as [Expose](https://github.com/vchoutas/expose), [VIBE](https://github.com/mkocabas/VIBE), [HuManiFlow](https://github.com/akashsengupta1997/HuManiFlow) can be an example of this field.

Despite the development of wonderful models, it is difficult to use them in the platforms that does not use Python language. Unity is one example of those.
The reason behind this is because, it is quite burdensome, or even impossible, to port them using onnx-barracuda.
Even if you are successful in changing them into a onnx model, accuracy drop is an inevitable problem. This inaccuracy is a serious problem when it comes to 'Real-time' 'accurate' pose/shape estimation.

So, what we've decided to do is : Using two separate models for shape reconstruction and pose estimation - which are executed in two different language, Python and C#(Unity).
> The project uses [SMPL](https://github.com/vchoutas/smplx) model to represent human body - which can be easily manipulated in Unity using Shape and Pose parameters.

> Shape estiamtion is done by a Python server. The client(Unity side) requests for shape estimation using current frame's RGB image by sending the server byte-converted image. After the estimation, the server sends the result of estimation in the format of shape parameters back to the client. This process is done asynchronously.
> > When the server receives the image data, it would crop and pad the image to a desired size in order to perform the inference.
> > We are using [HuManiFlow](https://github.com/akashsengupta1997/HuManiFlow) model for python server.

> Pose estimation is done in Unity, using [MediaPipe](https://developers.google.com/mediapipe). We've used [MediaPipeUnityPlugin](https://github.com/homuler/MediaPipeUnityPlugin), since there is a plugin that has already been developed for in-Unity use.
A big cheers to [homuler](https://ko-fi.com/homuler)(the author of the plugin).


--------------
# Architecture
Here is an image showing the overall architecture design of the project.


--------------
# Please Note that :
In order to do shape representation, a TCP server must be reconstructed using the code
