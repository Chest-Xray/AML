"use server"

async function uploadImage(formData: FormData) {
    const response = await fetch(
        "http://127.0.0.1:8000/prediction",
        {
            method: "POST",
            body: formData,
            headers: {
                contentType: "multipart/form-data", 
            },
        }
    );

    return response.json()
}

export default uploadImage;