// Talks to the Python server (app/guidance/serve_run.py).

export async function getJson(url) {
  const response = await fetch(url);
  return response.json();
}

// POSTs form data and returns the parsed JSON. On an error status it throws an Error whose message is
// the server's "detail" text, so the page can show it to the user.
export async function postForm(url, formData) {
  const response = await fetch(url, { method: 'POST', body: formData });
  if (!response.ok) {
    let message = response.statusText;
    try {
      message = (await response.json()).detail || message;
    } catch (e) {
      // the error body was not JSON; keep the status text
    }
    throw new Error(message);
  }
  return response.json();
}
