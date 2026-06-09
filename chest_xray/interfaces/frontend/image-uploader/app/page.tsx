"use client"
import { SubmitEvent, useState } from 'react';
import uploadImage from '../services/uploadImage';


export default function StartPage() {
  const [predictions, setPredictions] = useState<{ label: string; confidence: number }[]>([]);
  const [returnedImages, setReturnedImages] = useState<{ label: string; image: string }[]>([]);
  const [gradcamImages, setGradcamImages] = useState<{ label: string; image: string }[]>([]);

  const handleSubmit = async (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    console.log("Form submitted");
    if (!event.currentTarget.file.files || event.currentTarget.file.files.length === 0) {
      console.error("No file selected");
      return;
    }
    const formData = new FormData(event.currentTarget);

    try {
      const result = await uploadImage(formData);
      console.log("Upload result:", result);
      setPredictions(result.predictions);
        setReturnedImages(result.bbox_images || []);
      setGradcamImages(result.gradcam_images || []);
      console.log("Predictions set:", predictions);
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="upload-wrapper flex flex-col items-center gap-4">
        <h1 className="text-2xl font-bold">Upload your chest X-ray image</h1>
        <form className='flex flex-col' onSubmit={handleSubmit}>
          <input className="file-input" type="file" accept="image/*" name="file" />
          <button className="submit-button mt-4" type="submit">Upload</button>
        </form>
      </div>
      <div className="results flex w-full flex-col md:flex-row items-start gap-6 mt-4 max-w-4xl">
      {returnedImages.length > 0 && (
        <div className="returned-image mt-4">
          <h2 className="text-xl font-bold">Bounding Box Predictions:</h2>
          <ul className="bbox-list mt-2">
          {returnedImages.map((img, index) => (
              <li key={index} className="mb-4">
                <p className="text-lg mt-2">{img.label}</p>
                <img
                  src={`data:image/jpeg;base64,${img.image}`}
                  alt={img.label}
                  className="max-w-full h-auto border border-gray-700"
                />
              </li>
          ))}
          </ul>
        </div>
      )}
      {gradcamImages.length > 0 && (
        <div className="gradcam-images mt-4">
          <h2 className="text-xl font-bold">Grad-CAM Overlays:</h2>
          <ul className="bbox-list mt-2">
            {gradcamImages.map((g, idx) => (
              <li key={idx} className="mb-4">
                <p className="text-lg mt-2">{g.label}</p>
                <img src={`data:image/jpeg;base64,${g.image}`} alt={g.label} className="max-w-full h-auto border border-red-500" />
              </li>
            ))}
          </ul>
        </div>
      )}
      {predictions.length > 0 && (
        <div className="prediction-result mt-4">
          <h2 className="text-xl font-bold">Predictions:</h2>
          <table className="min-w-full bg-transparent text-white mt-2 border border-gray-700">
            <thead>
              <tr>
                <th className="px-4 py-2 text-left">Rank</th>
                <th className="px-4 py-2 text-left">Label</th>
                <th className="px-4 py-2 text-left">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((p, i) => (
                <tr key={i} className={i % 2 === 0 ? "bg-transparent" : "bg-black"}>
                  <td className="px-4 py-2">{i + 1}</td>
                  <td className="px-4 py-2">{p.label}</td>
                  <td className="px-4 py-2">{p.confidence.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      </div>
    </main>
  );
}
