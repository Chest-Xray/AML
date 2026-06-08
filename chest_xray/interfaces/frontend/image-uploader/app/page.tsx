"use client"
import { SubmitEvent, useState } from 'react';
import uploadImage from '../services/uploadImage';
import Image from 'next/image';


export default function StartPage() {
  const [predictions, setPredictions] = useState<{ label: string; confidence: number }[]>([]);
  const [returnedImage, setReturnedImage] = useState<string>("");

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
      setReturnedImage(result.image);
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
      {returnedImage && (
        <div className="returned-image mt-4">
          <h2 className="text-xl font-bold">Uploaded Image:</h2>
          <Image src={`data:image/png;base64,${returnedImage}`} alt="Uploaded" className="max-w-full h-auto" width={500} height={500} />
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
