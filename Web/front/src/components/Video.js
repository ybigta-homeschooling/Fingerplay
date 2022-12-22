import "./components.css";
function Video({ Id }) {
  const videoData = {
    1: "rudolph.mov",
    2: "bear.mov",
    3: "rabbit.mov",
    4: "butterfly.mov",
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
