import "./components.css";
function Video({ Id }) {
  const videoData = {
    1: "rudolph1.mov",
    2: "bear1.mov",
    3: "rabbit1.mov",
    4: "butterfly1.mov",
  };
  return (
    <div className="video">
      <img src="/video_background.png" />
      <video controls="controls" autoPlay>
        <source src={require(`../video/${videoData[Id]}`)} type="video/mp4" />
      </video>
    </div>
  );
}

export default Video;
