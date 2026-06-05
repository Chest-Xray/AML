"use client"
import { SubmitEvent, useState } from 'react';
import uploadImage from '../services/uploadImage';



export default function StartPage() {
  const [predictions, setPredictions] = useState<{ label: string; confidence: number }[]>([]);

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
      {predictions.length > 0 && (
        <div className="prediction-result mt-4">
          <h2 className="text-xl font-bold">Predictions:</h2>
          <ul>
            {predictions.map((p, i) => (
              <li key={i}>
                {p.label} - {p.confidence.toFixed(3)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </main>
  );
}
