const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");

const chatResponse = document.getElementById("chat-response");
const answerText = document.getElementById("answer");
const citationsBadges = document.getElementById("citations");

const snippetsSection = document.getElementById("snippets-section");
const snippetsDiv = document.getElementById("snippets");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    resetUI();

    if (!question) {
        showError("Please enter a question.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });

        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error(`Server error (${response.status}). Please try again.`);
        }

        if (!response.ok) {
            throw new Error(data.error || "Request failed.");
        }

        displayAnswer(data.answer);
        displayCitations(data.citations || []);
        displaySnippets(data.snippets || []);
    } catch (error) {
        showError(error.message || "Something went wrong.");
    } finally {
        setLoading(false);
    }
});

function resetUI() {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";

    chatResponse.classList.add("hidden");
    snippetsSection.classList.add("hidden");

    answerText.textContent = "";
    citationsBadges.innerHTML = "";
    snippetsDiv.innerHTML = "";
}

function setLoading(isLoading) {
    loading.classList.toggle("hidden", !isLoading);
    form.querySelector("button").disabled = isLoading;
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function displayAnswer(answer) {
    answerText.textContent = answer;
    chatResponse.classList.remove("hidden");
}

function displayCitations(citations) {
    if (citations.length === 0) return;

    citationsBadges.innerHTML = "";

    citations.forEach((citation) => {
        const badge = document.createElement("span");
        badge.className = "citation-badge";
        badge.textContent = citation.document_title;
        if (citation.page) badge.textContent += ` p.${citation.page}`;
        citationsBadges.appendChild(badge);
    });
}

function displaySnippets(snippets) {
    if (snippets.length === 0) return;

    snippetsDiv.innerHTML = "";

    snippets.forEach((snippet) => {
        const card = document.createElement("div");
        card.className = "snippet-card";

        const meta = document.createElement("div");
        meta.className = "snippet-meta";
        meta.textContent = snippet.document_title;

        const text = document.createElement("p");
        text.textContent = snippet.snippet;

        card.appendChild(meta);
        card.appendChild(text);
        snippetsDiv.appendChild(card);
    });

    snippetsSection.classList.remove("hidden");
}
