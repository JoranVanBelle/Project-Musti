let opnieuw = window.localStorage.getItem("opnieuw");

numberOfFiles = window.localStorage.getItem("numberOfFiles")

function choosePicture() {
  let randomValue = Math.floor(Math.random() * numberOfFiles);
  localStorage.setItem("img", randomValue)
}

document.getElementById("newPicture").onclick = () => {
  choosePicture();
  window.location.href = `/picture?image=${localStorage.getItem("img")}`;
}

document.getElementById("retrain").onclick = () => {
  document.getElementById("title").innerText = "Retraining the model. This can take a moment, please be patient...";
  document.getElementById("container").innerHTML="";
  window.location.href = "/retraining";
  console.log("retraining");
}
