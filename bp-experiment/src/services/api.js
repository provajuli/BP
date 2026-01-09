export async function submitResult(payload) {
    const res = await fetch("api/submit.php", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    const json = await res.json().catch(() => ({}));
    if (!res.ok || !json.ok) throw new Error(json.error || `Submit failed (${res.status})`);
    return json;
}
