## Objective

1. Use **FastAPI** to create an API for your model
2. Run that API on your machine
3. Put it in production on the cloud

## Warning

This challenge will require a lot of downloading and uploading, especially in parts 4 and 5. Make sure you have a stable and fast internet connection, and are not on a limited plan. Don't do parts 4 and 5 of this challenge using your mobile phone's network.

## Context

Now that we have a performant model trained in the cloud, we will expose it to the world 🌍

We will create a **prediction API** for our model, run it on our machine to make sure that everything works correctly, and then we will deploy it in the cloud so that everyone can play with our model!

To do so, we will:

👉 create a **prediction API** using **FastAPI**

👉 create a **Docker image** containing the environment required to run the code of our API

👉 push this image to **Google Cloud Run** so that it runs inside a **Docker container** that will allow developers all over the world to use it

# 1️⃣ Project Setup 🛠

<details>
  <summary markdown='span'><strong>❓Instructions </strong></summary>

## Environment

Copy your `.env` file from the previous package version:

```bash
cp ~/code/<user.github_nickname>/{{local_path_to('07-ML-Ops/03-Automate-model-lifecycle/01-Automate-model-lifecycle')}}/.env .env
```

OR

Use the provided `env.sample`, replacing the environment variable values with yours.

## API Directory

A new `taxifare/api` directory has been added to the project to contain the code of the API along with 2 new configuration files, which can be found in your project's root directory:

```bash
.
├── Dockerfile          # 🎁 NEW: building instructions
├── Makefile            # good old manual task manager
├── README.md
├── requirements.txt    # all the dependencies you need to run the package
├── setup.py
├── taxifare
│   ├── api             # 🎁 NEW: API directory
│   │   ├── __init__.py
│   │   └── fast.py     # 🎁 NEW: where the API lives
│   ├── interface       # package entry point
│   └── ml_logic
└── tests
```

Now, have a look at the `requirements.txt`. You can see newcomers:

``` bash
# API
fastapi         # API framework
pytz            # time zone management
uvicorn         # web server
# tests
httpx           # HTTP client
pytest-asyncio  # asynchronous I/O support for pytest
```

⚠️ Make sure to perform a **clean installation** of the package.

<details>
  <summary markdown='span'>❓How?</summary>

`make reinstall_package`, of course 😉

</details>

## Running the API with FastAPI and a Uvicorn Server

We provide you with a FastAPI skeleton in the `fast.py` file.

**💻 Try to launch the API now!**

<details>
  <summary markdown='span'>💡 Hint</summary>

You probably want a `uvicorn` web server with 🔥 hot-reloading...

In case you can't find the proper syntax, look at your `Makefile`: we provided you with a new task: `run_api`.

<details>
<summary markdown='span'>Error <code>Address already in use</code>?</summary>

If you run into the error `Address already in use`, the port `8000` on your local machine might already be occupied by another application.

You can check this by running `lsof -i :8000`. If the command returns something, then port `8000` is already in use.

In this case, you have three options:
1. Look in your terminal windows to check if another process is running on port 8000 (maybe another `uvicorn` instance?).
2. Use the `kill` command followed by the `PID` of the process currently using port `8000` (you will find the `PID` in the second column of the output of the `lsof` command you just ran), e.g. `kill 1234`.
3. Specify another port in the [0, 65535] range in the `run_api` command using the `--port` parameter.

</details>

</details>
<br>

**❓ How do you consult your running API?**

<details>
  <summary markdown='span'>Answer</summary>

💡 Your API is available locally on port `8000`, unless otherwise specified 👉 [http://localhost:8000](http://localhost:8000) or [http://127.0.0.1:8000](http://127.0.0.1:8000)

These two addresses are equivalent: `localhost` or `127.0.0.1` is the address of your own machine.

Go visit it!
</details>

You probably don't see much ... yet!

**❓ Which endpoints are available?**

<details>
  <summary markdown='span'>Answer</summary>

There is only one endpoint (_partially_) implemented at the moment, the root endpoint `/`.
The "unimplemented" root page is a little raw, but remember that you can always find more info on the API using the Swagger endpoint 👉 [http://localhost:8000/docs](http://localhost:8000/docs)

</details>

</details>


# 2️⃣  Build the API 📡

<details>
  <summary markdown='span'><strong>❓Instructions </strong></summary>
An API is defined by its specifications (for a complete example see [GitHub repositories API](https://docs.github.com/en/rest/repos/repos)). Below you will find the API specifications you need to implement.

## Specifications

### Root

- Denoted by the `/` character
- HTTP verb: `GET`

In order to easily test your `root` endpoint, use the following response example as a goal:
```json
{
    'greeting': 'Hello'
}
```

- 💻 Implement the **`root`** endpoint `/`
- 👀 Look at your browser 👉 **[http://localhost:8000](http://localhost:8000)**
- 🐛 Inspect the server logs and, if needed, add some **`breakpoint()`s** to debug

When and **only when** your API responds as required:
1. 🧪 Test your implementation with `make test_api_root`
2. 🧪 Track your progress on Kitt with  `make test_kitt
3. 🧪 Add, commit and push your code! (Make sure to include the `tests/api/test_output.txt` file to track your progress)

### Prediction

- Denoted by `/predict`
- HTTP verb: `GET`

It should accept the following query parameters

<br>

| Name | Type | Sample |
|---|---|---|
| pickup_datetime | DateTime | `2014-07-06 19:18:00` |
| pickup_longitude | float | `-73.950655` |
| pickup_latitude | float | `40.783282` |
| dropoff_longitude | float | `-73.984365` |
| dropoff_latitude | float | `40.769802` |
| passenger_count | int | `2` |

<br>

It should return the following JSON:
```json
{
    'fare': 14.710600943237969
}
```

**❓ How would you proceed to implement the `/predict` endpoint? Discuss with your buddy 💬**

Ask yourselves the following questions:
- How should we build `X_pred`? How to handle timezones ?
- How can we reuse the `taxifare` model package in the most lightweight way ?
- How to render the correct response?

<details>
  <summary markdown='span'>💡 Hints</summary>

- Re-use the methods available in the `taxifare/ml_logic` package rather than the main routes in `taxifare/interface`; always load the minimum amount of code possible!

</details>


👀 Inspect the **response** in your **browser**, and inspect the **server logs** while you're at it

👉 Call the API in your browser with this example: [http://localhost:8000/predict?pickup_datetime=2014-07-06%2019:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2](http://localhost:8000/predict?pickup_datetime=2014-07-06%2019:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2)

👉 Or call the API from your CLI

```bash
curl -X 'GET' \
  'http://localhost:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2' \
  -H 'accept: application/json'
```

When and **only when** your API responds as required:
1. 🧪 Test your implementation with `make test_api_predict`
2. 🧪 Track your progress on Kitt with  `make test_kitt`
3. 🧪 Add, commit and push your code! (Make sure to include the `tests/api/test_output.txt` file to track your progress)

**👏 Congrats, you have built your first ML predictive API!**

<br>

### ⚡️ Faster Predictions

Did you notice your predictions were a bit slow? Why do you think that is?

The answer is visible in your logs!

We want to avoid loading the heavy Deep Learning model from MLflow at each `GET` request! The trick is to load the model into memory on startup and store it in a global variable in `app.state`, which is kept in memory and accessible across all routes!

This will prove very useful for Demo Days!

<details>
  <summary markdown='span'>⚡️ like this ⚡️</summary>

```python
app = FastAPI()
app.state.model = ...

@app.get("/predict")
...
app.state.model.predict(...)
```

</details>

👉 Now, stop the `uvicorn` server that is running in your terminal. Use `Ctrl-C` to stop it.

</details>


# 3️⃣ Build a Docker Image for our API 🐳

<details>
  <summary markdown='span'><strong>❓ Instructions </strong></summary>

We now have a working **predictive API** that can be queried from our local machine.

We want to make it available to the world. To do that, the first step is to create a **Docker image** that contains the environment required to run the API. We will then run it _locally_ using Docker on your own machine to test if everything still works.

**❓ What are the 3 steps to run the API on Docker?**

<details>
  <summary markdown='span'>Answer</summary>

1. **Create** a `Dockerfile` containing the instructions to build the API
2. **Build** the image
3. **Run** the API on Docker (locally) to ensure that it is responding as required

</details>

## 3.1) Setup

You need to have the Docker daemon running on your machine to be able to build and run the image.

**💻 Launch Docker Daemon**

<details>
  <summary markdown='span'>macOS</summary>

Launch the Docker app, you should see a whale in your menu bar.

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/macos-docker-desktop-running.png" target="_blank"><img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/macos-docker-desktop-running.png" width="150" alt="verify that Docker Desktop is running"></a>

</details>

<details>
  <summary markdown='span'>Windows WSL2</summary>

Launch the Docker app, you should see a whale in your taskbar (Windows).

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/windows-docker-app.png" target="_blank"><img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/windows-docker-app.png" width="150" alt="verify that Docker Desktop is running"></a>

</details>

<details>
  <summary markdown='span'>Ubuntu</summary>

Nothing to do. Docker should work out of the box.

</details>

**✅ Check whether the Docker daemon is up and running with `docker info` in your Terminal**

A nice stack of logs should print:
<br>
<a href="https://github.com/lewagon/data-setup/raw/master/images/docker_info.png" target="_blank"><img src='https://github.com/lewagon/data-setup/raw/master/images/docker_info.png' width=150></a>

<details>
  <summary markdown='span'>Permission denied? (WSL / Ubuntu)</summary>

Run the following commands one by one:

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

Try `docker info` again.

Seeing this error?
```
WARNING: Error loading config file: /home/user/.docker/config.json - stat /home/user/.docker/config.json: permission denied`?
```

Run the following command:

```bash
sudo rm -rf ~/.docker/
```

Try `docker info` again.

</details>


## 3.2) `Dockerfile`

As a reminder, here is the project directory structure:

```bash
.
├── Dockerfile          # 🆕 Building instructions
├── Makefile
├── README.md
├── requirements.txt    # All the dependencies you need to run the package
├── setup.py            # Package installer
├── taxifare
│   ├── api
│   │   ├── __init__.py
│   │   └── fast.py     # ✅ Where the API lays
│   ├── interface       # Manual entry points
│   └── ml_logic
└── tests
```

**❓ What are the key ingredients a `Dockerfile` needs to cook a delicious Docker image?**

<details>
  <summary markdown='span'>Answer</summary>

Here are the most common instructions for any good `Dockerfile`:
- `FROM`: select a base image for our image (the environment in which we will run our code), this is usually the first instruction
- `COPY`: copy files and directories into our image (our package and the associated files, for example)
- `RUN`: execute a command **inside** of the image during the building process (for example, `pip install -r requirements.txt` to install package dependencies)
- `CMD`: the **main** command that will be executed when we run our **Docker image**. There can only be one `CMD` instruction in a `Dockerfile`. It is usually the last instruction!

</details>

**❓ What should the base image contain so we can build our image on top of it?**

<details>
  <summary markdown='span'>💡 Hints</summary>

You can start from a raw Linux (Ubuntu) image, but then you'll have to install Python and `pip` before installing `taxifare`!

OR

You can choose an image with Python (and pip) already installed! (recommended) ✅

</details>

**💻 In the `Dockerfile`, write the instructions needed to build the API image following these specifications:** <br>
_Feel free to use the checkboxes below to help you keep track of what you've already done_ 😉


The image should contain:
<br>
<input type="checkbox" id="dockertask1" name="dockertask1" style="margin-left: 20px;">
<label for="dockertask1"> the same Python version of your virtual env</label><br>
<input type="checkbox" id="dockertask2" name="dockertask2" style="margin-left: 20px;">
<label for="dockertask2"> all the directories from the `/taxifare` project needed to run the API</label><br>
<input type="checkbox" id="dockertask3" name="dockertask3" style="margin-left: 20px;">
<label for="dockertask3"> the list of dependencies (don't forget to install them!)</label><br>

The web server should:
<br>
<input type="checkbox" id="dockertask4" name="dockertask4" style="margin-left: 20px;">
<label for="dockertask4"> launch when a container is started from the image</label><br>
<input type="checkbox" id="dockertask5" name="dockertask5" style="margin-left: 20px;">
<label for="dockertask5"> listen to the HTTP requests coming from outside the container (see `host` parameter)</label><br>
<input type="checkbox" id="dockertask6" name="dockertask6" style="margin-left: 20px;">
<label for="dockertask6"> be able to listen to a specific port defined by an environment variable `$PORT` (see `port` parameter)</label><br>

<details>
  <summary markdown='span'>⚡️ Kickstart pack</summary>

Here is the skeleton of the `Dockerfile`:

  ```Dockerfile
  FROM image
  COPY taxifare
  COPY dependencies
  RUN install dependencies
  CMD launch API web server
  ```

</details>


**❓ How do you check if the `Dockerfile` instructions will execute what you want?**

<details>
  <summary markdown='span'>Answer</summary>

You can't at this point! 😁 You need to build the image and check if it contains everything required to run the API. Go to the next section: Build the API image.
</details>

## 3.3) Build the API image

Now is the time to **build** the API image so you can check if it satisfies all requirements, and to be able to run it on Docker.

**💻 Choose a Docker image name and add it to your `.env`**.
You will be able to reuse it in the `docker` commands:

``` bash
GAR_IMAGE=taxifare
```

**💻 Then, make sure you are in the directory of the `Dockerfile` and build `.`** :

First hit return on your keyboard to make sure the variables from your `.env` are loaded. (You could also write `direnv reload`, but a simple return is enough thanks to our setup.)

```bash
docker build --tag=$GAR_IMAGE:dev .
```


**💻 Once built, the image should be visible in the list of images built with the following command**:

``` bash
docker images
```
<img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/docker_images.png'>

🤔 The image you are looking for does not appear in the list? Ask for help 🙋‍♂️

## 3.4) Check the API Image

Now that the image is built, let's verify that it satisfies the specifications to run the predictive API. Docker comes with a handy command to **interactively** communicate with the shell of the image:

``` bash
docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev bash
```

<details>
  <summary markdown='span'>🤖 Command composition</summary>

- `docker run $GAR_IMAGE`: run the image
- `-it`: enable the interactive mode
- `-e PORT=8000`: specify the environment variable `$PORT` to which the image should listen
- `bash`: launch a shell console
</details>

A shell console should open, you are now inside the image 👏

**💻 Verify that the image is correctly set up:**

<input type="checkbox" id="dockertask7" name="dockertask7" style="margin-left: 20px;">
<label for="dockertask7"> The python version is the same as in your virtual env</label><br>
<input type="checkbox" id="dockertask8" name="dockertask8" style="margin-left: 20px;">
<label for="dockertask8"> The <code>/taxifare</code> directory exists</label><br>
<input type="checkbox" id="dockertask9" name="dockertask9" style="margin-left: 20px;">
<label for="dockertask9"> The <code>requirements.txt</code> file exists</label><br>
<input type="checkbox" id="dockertask10" name="dockertask10" style="margin-left: 20px;">
<label for="dockertask10"> The dependencies are all installed</label><br>

<details>
  <summary markdown='span'>🙈 Solution</summary>

- `python --version` to check the Python version
- `ls` to check the presence of the files and directories
- `pip list` or `pip freeze` to check if requirements are installed
</details>

Exit the terminal and stop the container at any moment with:

``` bash
exit
```

**✅ ❌ All good? If something is missing, you will probably need to fix your `Dockerfile` and re-build the image**

## 3.5) Run the API Image

In the previous section you learned how to interact with the shell inside the image. Now is the time to run the predictive API image and test if the API responds as it should.

**💻 Try to actually run the image**

You want to `docker run ...` without the `bash` command at the end. This will trigger the `CMD` line of your Dockerfile, instead of just opening a shell.

``` bash
docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev
```

**😱 It is probably crashing with errors involving environment variables**

**❓ What's wrong? What's the difference between your local environment and your image environment? 💬 Discuss with your buddy.**

<details>
  <summary markdown='span'>Answer</summary>

There is **no** `.env` in the image! The image has **no** access to the environment variables 😈
</details>

**💻 Adapt the run command so the `.env` is sent to the image (use `docker run --help` to help you!)**

<details>
  <summary markdown='span'>🙈 Solution</summary>

`--env-file` to the rescue!

```bash
docker run -it -e PORT=8000 -p 8000:8000 --env-file your/path/to/.env $GAR_IMAGE:dev
```
</details>

**❓ How would you check that the image runs correctly?**

<details>
  <summary markdown='span'>💡 Hints</summary>

The API should respond in your browser, go visit it!

Also, you can check if the image runs with `docker ps` in a new Terminal tab or window.

Your browser doesn't respond? Make sure you visited the right address: in your terminal, you will see `uvicorn` telling you it is running on `0.0.0.0:8000`, but that is inside the container. We still need to access the container from our local machine, and our local machine's address is `127.0.0.1`, or `localhost`. So visit `http://localhost:8000` or `http://127.0.0.1:8000`.

Error `Address already in use`? You probably still have `uvicorn` running somewhere. We provided you with instructions to solve this in part **1 - Project Setup** of this challenge.

</details>


### It's alive! 😱 🎉

<br>

**👀 Inspect your browser response 👉 [http://localhost:8000/predict?pickup_datetime=2014-07-06&19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2](http://localhost:8000/predict?pickup_datetime=2014-07-06&19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2)**

When and **only when** your API responds as required:
1. 🧪 Test your implementation with `make test_api_on_docker`
2. 🧪 Track your progress on Kitt with  `make test_kitt`
3. 🧪 Add, commit and push your code! (Make sure to include the `tests/api/test_output.txt` file to track your progress)


**🛑 You can stop your container with `docker container stop <CONTAINER_ID>`**<br>👉 To find the container id: run `docker ps` in another terminal window or tab


👏 Congrats, you've built your first ML predictive API inside a Docker container!

<br>


</details>


# 4️⃣ Deploy the API 🌎

<details>
  <summary markdown='span'><strong>❓Instructions </strong></summary>

Now that we have built a **predictive API** Docker image that we can run on our local machine, we are 2 steps away from deploying; we just need to:
1. push the **Docker image** to **Google Artifact Registry**
2. deploy the image on **Google Cloud Run** so that it gets instantiated into a **Docker container**

## 4.1) Push our prod image to Google Artifact Registry

**❓What is the purpose of Google Artifact Registry?**

<details>
  <summary markdown='span'>Answer</summary>

**Google Artifact Registry** is a cloud storage service for Docker (and similar technology) images with the purpose of allowing **Cloud Run** or **Kubernetes Engine** to serve them.

It is, in a way, similar to **GitHub** allowing you to store your git repositories in the cloud — except Google Artifact Registry lacks a dedicated user interface and additional services such as `forks` and `pull requests`).

</details>


### Build and Push the Image to GAR

👏 Everything runs fine on your local machine. Great. We will now deploy your image on servers that are going to run these containers online for you.

However, note that these servers (Google Cloud Run servers) will be running on **AMD/Intel x86 processors**, not ARM/M1, as most cloud providers still run on Intel.

<details>
  <summary markdown='span'><strong>🚨 If you have Mac Silicon (M-chips) or ARM CPU, read carefully</strong></summary>

Because your processor is very different from the ones used by Cloud Run, a docker image built for your machine will not work on Cloud Run. And an image built for Cloud Run will not work on your machine...

The solution is to use one image to test your code locally (you have just done it above), and another one to push your code to production. We will do that in the next steps.

</details>

Now we are going to build our image again in a way we can use it on Cloud Run. If you have an Intel/AMD processor, this should be pretty fast since Docker is smart and is going to reuse all the building blocks that were previously used to build the prediction API image.

First, let's make sure to enable the [Google Artifact Registry API](https://console.cloud.google.com/flows/enableapi?apiid=artifactregistry.googleapis.com&redirect=https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images) for your project in GCP.

Once this is done, let's allow the `docker` command to push an image to GCP within our region.

``` bash
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
```

Let's create a repo in that region as well!

```bash
gcloud artifacts repositories create taxifare --repository-format=docker \
--location=$GCP_REGION --description="Repository for storing taxifare images"
```

Let's build our image ready to push to that repo.

``` bash
docker build \
  --platform linux/amd64 \
  -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/taxifare/$GAR_IMAGE:prod .
```

See how we tag it with `prod` this time. This is the version we'll run on Cloud Run.

<details>
  <summary markdown='span'>🔎 Wonder what the <code>--platform linux/amd64</code> does?
  </summary>

  We use the `--platform linux/amd64` argument to make sure we build an image for the architecture that Cloud Run is using. We need this if you are using ARM processor based machines like MacBooks on Apple Silicon (M-chips). You don't really need this on Windows / Linux / Apple machines with Intel or AMD chips, but it doesn't hurt either.

</details>

We can now push our image to Google Artifact Registry.

``` bash
docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/taxifare/$GAR_IMAGE:prod
```

The image should be visible in the [GCP console](https://console.cloud.google.com/artifacts/).

## 4.2) Deploy the Artifact Registry Image to Google Cloud Run

Add a `--memory` flag to your project configuration and set it to `2Gi` (use `GAR_MEMORY` in `.env`)

👉 This will allow your container to run with **2GiB (= [Gibibyte](https://simple.wikipedia.org/wiki/Gibibyte))** of memory

**❓ How does Cloud Run know the values of the environment variables to be passed to your container? Discuss with your buddy 💬**

<details>
  <summary markdown='span'>Answer</summary>

It doesn't. You need to provide a list of environment variables to your container when you deploy it 😈

</details>

**💻 Using the `gcloud run deploy --help` documentation, identify a parameter that allows you to pass environment variables to your container on deployment**

<details>
  <summary markdown='span'>Answer</summary>

The `--env-vars-file` is the correct one!

```bash
gcloud run deploy --env-vars-file .env.yaml
```

<br>
Tough luck, the <code>--env-vars-file</code> parameter takes as input the name of a YAML (pronounced "yemil") file containing the list of environment variables to be passed to the container.

</details>

**💻 Create a `.env.yaml` file containing all the necessary environment variables**

You can use the provided `.env.sample.yaml` file as a source for the syntax (do not forget to update the values of the parameters). All values should be strings

**❓ What is the purpose of Cloud Run?**

<details>
  <summary markdown='span'>Answer</summary>

Cloud Run will instantiate the image into a container and run the `CMD` instruction inside of the `Dockerfile` of the image. This last step will start the `uvicorn` server, thus serving our **predictive API** to the world 🌍

</details>

Let's run one last command 🤞

``` bash
gcloud run deploy --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/taxifare/$GAR_IMAGE:prod --memory $GAR_MEMORY --region $GCP_REGION --env-vars-file .env.yaml
```

After confirmation, you should see something like this, indicating that the service is live 🎉

```bash
Service name (wagon-data-tpl-image):
Allow unauthenticated invocations to [wagon-data-tpl-image] (y/N)?  y

Deploying container to Cloud Run service [wagon-data-tpl-image] in project [le-wagon-data] region [europe-west1]
✓ Deploying new service... Done.
  ✓ Creating Revision... Revision deployment finished. Waiting for health check to begin.
  ✓ Routing traffic...
  ✓ Setting IAM Policy...
Done.
Service [wagon-data-tpl-image] revision [wagon-data-tpl-image-00001-kup] has been deployed and is serving 100 percent of traffic.
Service URL: https://wagon-data-tpl-image-xi54eseqrq-ew.a.run.app
```

Copy the service URL, and check in your browser that it works.


When and **only when** your API responds as required:

1. 🧪 Write down your service URL in your local `.env` file so we can test it!
    ```bash
    SERVICE_URL=https://wagon-data-tpl-image-xi54eseqrq-ew.a.run.app
    ```

2. Reload direnv in the terminal: `direnv reload`
3. 🧪 Test your implementation with `make test_api_on_prod`
4. 🧪 Track your progress on Kitt with  `make test_kitt`
5. 🧪 Add, commit and push your code! (Make sure to include the `tests/api/test_output.txt` file to track your progress)


**👏👏👏👏 MASSIVE CONGRATS 👏👏👏**
You deployed your first ML predictive API!
Any developer in the world 🌍 is now able to browse to the deployed url and get a prediction using the API 🤖!

<br>

## 4.3) Stop everything and save resources 💸

⚠️ **Only do this after the next module!** In the next module we will make a front end and connect it to our own API. So come back here after the next module's challenges.

### Cloud Run

💸 By default you only pay for Cloud Run when it is actually doing something: starting up, shutting down and handling requests. You do not pay for the so-called idle time, when the instances are doing nothing, or are shut down, **unless** you set minimum instances higher than the default 0. 💸

You can look for any running cloud run services using

``` bash
gcloud run services list
```

You can shut down any instance with

``` bash
gcloud run services delete $INSTANCE
```
### Your local machine

You can also stop (or kill) your local docker image to free up memory on your local machine

``` bash
docker stop 152e5b79177b  # ⚠️ use the correct CONTAINER ID
docker kill 152e5b79177b  # ☢️ only if the image refuses to stop (did someone create an ∞ loop?)
```
Remember to stop the Docker daemon in order to free resources on your machine once you are done using it.

<details>
  <summary markdown='span'>macOS</summary>

Stop the `Docker.app` by clicking on **whale > Quit Docker Desktop** in the menu bar.
</details>

<details>
  <summary markdown='span'>Windows WSL2/Ubuntu</summary>

Stop the Docker app by right-clicking the whale on your taskbar.
</details>

</details>


# 5️⃣ OPTIONAL

If you have made it this far, expand your docker and API skills to:

1. Optimize your docker image
2. Handle POST requests for batch predictions
3. Handle POST requests to work with images

<details>
  <summary markdown='span'><strong>❓ Instructions </strong></summary>

## 1) Optimize your docker image

You probably noticed that these docker images become quite large, and that it takes quite some time to build them.

There are a couple of ways you can improve this process. Let's have a look at them.

### 1.1) Smarter image 🧠

**🤔 How do you avoid rebuilding all pip dependencies each time taxifare code is changed?**

<details>
  <summary markdown='span'>🎁 Solution</summary>

By leveraging Docker caching layer per layer. If you don't update a deeper layer, docker will not rebuild it!

Look at this non-optimized Dockerfile:

```Dockerfile
FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
COPY taxifare taxifare

RUN pip install -r requirements.txt

# ...
```

And compare it with the optimized version:

```Dockerfile
FROM python:3.10.6-buster

WORKDIR /prod

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Only then copy taxifare!
COPY taxifare taxifare

# ...
```

In the first version, each time you make a change to your taxifare code, the `RUN pip install -r requirements.txt` layer will be rebuild, because it comes after the change.

In the second one, we first install the requirements, and then copy our own code. As long as your `requirements.txt` doesn't change, the `RUN pip install -r requirements.txt` layer will not have to be rebuild.

</details>


### 1.2) Lighter image 🪶

As a responsible ML Engineer, you know that the size of an image is important when it comes to production. Depending on the base image you used in your `Dockerfile`, the API image could be huge:
- `python:3.10.6-buster` 👉 `4.1GB`
- `python:3.10.6-slim`   👉 `3.4GB`

The difference between these base images comes from the packages that are installed in the base Linux system. The `slim` base image comes with less packages, but is still fully functional.

There are even smaller `alpine` base images, but these are so barebones that you'll have to spend a lot of effort to make it work.

<details>
  <summary markdown='span'>Instructions</summary>

If you want to build using a `slim` image:
- In the `Dockerfile`, change the `FROM ...-buster` into `FROM ...-slim`.
- Specify another tag (e.g. `slim`) in your `docker build` command.

</details>

### 1.3) Ready-made base image

The base images we used so far contain a Linux distribution and a Python version.
An alternative is to use a base image that already contains your most important Python packages.

**❓ What is the heaviest requirement used by your API?**

<details>
  <summary markdown='span'>Answer</summary>

No doubt it is `tensorflow` with 1.1GB! Let's find a base image that is already optimized for it.
</details>

**📝 Change your base image**

<details>
  <summary markdown='span'>Instructions</summary>

Let's use a [TensorFlow docker image](https://hub.docker.com/r/tensorflow/tensorflow) instead! It's an Ubuntu Linux distribution with Python and TensorFlow already installed!

- 💻 Update your `Dockerfile` base image with either `tensorflow/tensorflow:2.10.0` (if you are on an Intel processor only)
- 💻 Remove `tensorflow` from your `requirements.txt` because it is now pre-built with the image.
- 💻 Build a lightweight local image of your API (you can use a tag:'light' on this new image to differentiate it from the heavy one built previously: `docker build --tag=$GAR_IMAGE:light .`
- 👀 Inspect the space saved with `docker images` and feel happy
- Also notice that the step where we `pip install` our packages goes much faster now, because we don't have to install TensorFlow anymore.

If your machine has an Intel/AMD processor, run the image, and check that the API still works.

If your system uses Apple Silicon (M-chips) or other ARM processors, you won't be able to run this image on your machine, only on Cloud Run. Base images with TensorFlow for Apple Silicon or other ARM processors also exist, but they are huge (over 7 Gb), so let's skip that.

</details>

### 1.4) Remove unnecessary packages

Have a look at the `requirements.txt` file. Do we need all these requirements in production?

Many of the packages we only needed in development or for training: matplotlib, seaborn, ipykernel, pytest, prefect.

Create `requirements_prod.txt` (a streamlined version of `requirements.txt`) by removing anything in `requirements.txt` that you will not need in production.

Change your Dockerfile to `COPY requirements_prod.txt requirements.txt` so that it uses this lighter requirements list.

This will save around 100 Mb, AND it will speed up the `pip install` step of the build.

### 1.5) Remove unnecessary caching files

When we `pip install`, pip automatically caches the files it downloads, to speed up the process next time you install the same packages.

Now, when we build a docker image, we start from scratch each time we run the `docker build`. So those caching files have no use, but they do take up a lot of space inside our image: up to 700 Mb for our api.

To avoid caching these files, we can use `pip install --no-cache-dir -r requirements.txt` in the Dockerfile.

### 1.6) Use a CPU only version of TensorFlow

By default the TensorFlow version has support to run on GPUs. In our docker container, we will only be doing "inference", making predictions, not training. Our task is thus pretty lightweight, and we can do without GPU support and install a CPU version of TensorFlow.

To do this:
- Start from a `slim` base image (not from the pre-built TensorFlow base image)
- In the `requirements_prod.txt`, change `tensorflow==2.10.0` into `tensorflow-cpu==2.10.0`.

This will save another 560 Mb. This image will be even smaller than the one using the pre-built TensorFlow image.

You can only build this for `--platform linux/amd64`. So if you're using an Apple Silicon (M-chip) or other ARM processor, you won't be able to run this image on your local machine, but you can run it on Cloud Run.

**Through a combination of all this steps, we can reduce the image size from over 4 Gb to less than 2 Gb! 🚀**

### 1.7) Prod-ready image (finally!) ☁️

Choose the combination of your liking, and make a new production ready image, tag it with `prod-light` and try to deploy it again.

<details>
  <summary markdown='span'>Option 1: start from a TensorFlow base image</summary>

- Use the TensorFlow image in your `Dockerfile`s `FROM`
- Remove tensorflow and all unnecessary packages from your `requirements_prod.txt`
- Use the `--no-cache-dir` when `pip install`ing

</details>

<details>
  <summary markdown='span'>Option 2: start from a <code>slim</code> base image</summary>

- Use the `slim` image in your `Dockerfile`s `FROM`
- Remove all unnecessary packages from your `requirements_prod.txt`
- Make sure `tensorflow-cpu` is included in your `requirements_prod.txt`
- Use the `--no-cache-dir` when `pip install`ing

</details>



Now build your new image:

- Tell Docker to build the image specifically for Intel/AMD processors and give it a new tag: `prod-light`:  <br>`docker build --platform linux/amd64 -t $GAR_IMAGE:prod-light .`
- If you're on Apple Silicon or other ARM processors, you will **not** be able to run this image locally, but this is the one you will be able push online to the GCP servers!

Push the new image and deploy it again.


## 2) Create a /POST request to be able to return batch predictions

Let's look at our `/GET` route format

```bash
http://localhost:8000/predict?pickup_datetime=2014-07-06&19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
```

🤯 How would you send a prediction request for 1000 rows at once?

The URL query string (everything after `?` in the URL above) is not able to send a large volume of data.

### Welcome to `/POST` HTTP Requests

- Your goal is to be able to send a batch of 1000 new predictions at once!
- Try to read more about POST in the [FastAPI docs](https://fastapi.tiangolo.com/tutorial/body/#request-body-path-query-parameters), and implement it in your package

## 3) Read about sending images 📸 via /POST requests to CNN models

In anticipation of your Demo Day, you might be wondering how to send unstructured data like images (or videos, sounds, etc.) to your Deep Learning model in prod.


👉 Bookmark [Le Wagon - data-template](https://github.com/lewagon/data-templates), and try to understand & reproduce the project boilerplate called "[sending-images-streamlit-fastapi](https://github.com/lewagon/data-templates/tree/main/project-boilerplates/sending-images-streamlit-fastapi)"


</details>
