## PYTHON (ML)

fastapi + htmx

```python
pip install fastapi uvicorn jinja2 python-multipart
```

tf2onnx 설치

```python
pip install -U tf2onnx onnxruntime
```

tensorflow 설치

```python
# Requires the latest pip
pip install --upgrade pip

# Current stable release for CPU and GPU
pip install tensorflow

# Or try the preview build (unstable)
pip install tf-nightly

# scikit-learn install
pip install scikit-learn
```

![mlp](https://github.com/maoli131/experimental-design/raw/main/images/MLP.png)

## HTMX

# 반환된 HTML을 DOM으로 스왑

와 마찬가지로 이 속성은 반환된 AJAX 응답이 DOM에 로드되는 방법을 정의하는 데 사용됩니다. 지원되는 값은 다음과 같습니다.hx-target, hx-swap

innerHTML: 기본값, 이 옵션은 요청을 보내는 현재 요소 내부에 AJAX 응답을 로드합니다.
outerHTML: 이 옵션은 요청을 보내는 전체 요소를 반환된 응답으로 바꿉니다
afterbegin: 요청을 보내는 요소의 첫 번째 자식으로 응답을 로드합니다.
beforebegin: 요청을 트리거하는 실제 요소의 부모 요소로 응답을 로드합니다.
beforeend: 요청을 보내는 요소의 마지막 자식 뒤에 AJAX 응답을 로드하고 추가합니다.
afterend: 이전과 달리 요청을 보내는 요소 뒤에 AJAX 응답을 추가합니다
none: 이 옵션은 AJAX 요청의 응답을 추가하거나 앞에 추가하지 않습니다

요청 표시기

```htmx
<div hx-get="http://path/to/api">
     <button>Click Me!</button>
     <img
        class="htmx-indicator"
        src="path/to/spinner.gif"
      />
</div>
```
