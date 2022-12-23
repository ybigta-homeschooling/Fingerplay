export default class APIService {
  // Insert an article
  static InsertArticle(animalId, level) {
    return fetch("http://localhost:5003/video_feed/" + animalId + "/" + level, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ score: -1, wa: [] }),
    })
      .then((response) => response.json())
      .catch((error) => console.log(error));
  }
}
