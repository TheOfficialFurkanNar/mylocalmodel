let currentConversationId = null;
let isWorking = false;

const messagesContainer = document.getElementById('messages');
const conversationList = document.getElementById('conversationList');
const conversationTitle = document.getElementById('activeConversationTitle');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');

document.getElementById('newConversationButton').addEventListener('click', startNewConversation);

chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = messageInput.value.trim();
    if (!question || isWorking) return;

    addUserMessage(question);
    messageInput.value = '';
    const assistantId = `assistant-${Date.now()}`;
    renderAssistantShell(assistantId);
    addLoadingIndicator(assistantId);

    isWorking = true;
    sendButton.disabled = true;
    messageInput.disabled = true;
    try {
        await sendMessage(question, assistantId);
    } catch (error) {
        console.error('Chat error:', error);
        updateAssistantText(assistantId, `Error: ${error.message}`);
        removeLoadingIndicator(assistantId);
    } finally {
        isWorking = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
    }
});

async function initializeHistory() {
    try {
        const response = await fetch('/conversations');
        const data = await response.json();
        const items = data.conversations || [];
        renderConversationList(items);

        const savedId = localStorage.getItem('localqwen-current-conversation');
        const selected = items.find((item) => item.id === savedId) || items[0];
        if (selected) await openConversation(selected.id);
    } catch (error) {
        console.error('Could not load conversation history:', error);
    }
}

async function sendMessage(question, assistantId) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000);
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: question, conversation_id: currentConversationId }),
            signal: controller.signal
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        currentConversationId = data.conversation_id;
        localStorage.setItem('localqwen-current-conversation', currentConversationId);
        (data.metadata?.tool_calls || []).forEach((toolCall, index) => {
            renderToolCallCard(assistantId, index, toolCall);
        });
        updateAssistantText(assistantId, data.response || '');
        renderMetadataCard(assistantId, data.metadata || {});
        removeLoadingIndicator(assistantId);
        conversationTitle.textContent = question.slice(0, 58) || 'New chat';
        await refreshConversationList();
    } catch (error) {
        if (error.name === 'AbortError') throw new Error('Request timed out after 5 minutes');
        throw error;
    } finally {
        clearTimeout(timeoutId);
    }
}

async function refreshConversationList() {
    const response = await fetch('/conversations');
    const data = await response.json();
    renderConversationList(data.conversations || []);
}

function renderConversationList(conversations) {
    conversationList.replaceChildren();
    if (!conversations.length) {
        const empty = document.createElement('p');
        empty.className = 'history-empty';
        empty.textContent = 'Your chats will appear here.';
        conversationList.appendChild(empty);
        return;
    }

    conversations.forEach((conversation) => {
        const row = document.createElement('div');
        row.className = 'conversation-row';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'conversation-item';
        if (conversation.id === currentConversationId) button.classList.add('active');
        const title = document.createElement('span');
        title.className = 'conversation-item-title';
        title.textContent = conversation.title || 'New chat';
        const detail = document.createElement('span');
        detail.className = 'conversation-item-detail';
        detail.textContent = `${formatDate(conversation.updated_at)} · ${Math.ceil(conversation.message_count / 2)} turns`;
        button.append(title, detail);
        button.addEventListener('click', () => openConversation(conversation.id));

        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.className = 'conversation-delete';
        deleteButton.textContent = '×';
        deleteButton.setAttribute('aria-label', `Delete conversation: ${conversation.title || 'New chat'}`);
        deleteButton.title = 'Delete conversation';
        deleteButton.disabled = isWorking;
        deleteButton.addEventListener('click', () => deleteConversation(conversation));

        row.append(button, deleteButton);
        conversationList.appendChild(row);
    });
}

async function deleteConversation(conversation) {
    if (isWorking || !window.confirm(`Delete “${conversation.title || 'New chat'}”? This cannot be undone.`)) return;
    try {
        const response = await fetch(`/conversations/${encodeURIComponent(conversation.id)}`, { method: 'DELETE' });
        if (!response.ok) throw new Error(`Conversation could not be deleted (HTTP ${response.status})`);
        if (currentConversationId === conversation.id) startNewConversation();
        await refreshConversationList();
    } catch (error) {
        console.error('Could not delete conversation:', error);
        window.alert(error.message);
    }
}

async function openConversation(conversationId) {
    if (isWorking) return;
    try {
        const response = await fetch(`/conversations/${encodeURIComponent(conversationId)}`);
        if (!response.ok) throw new Error('Conversation could not be loaded');
        const conversation = await response.json();
        currentConversationId = conversation.id;
        localStorage.setItem('localqwen-current-conversation', currentConversationId);
        conversationTitle.textContent = conversation.title || 'New chat';
        messagesContainer.replaceChildren();

        for (const message of conversation.messages || []) {
            if (message.role === 'user') {
                addUserMessage(message.content, false);
            } else if (message.role === 'assistant') {
                const assistantId = message.id || `assistant-${Date.now()}-${Math.random()}`;
                renderAssistantShell(assistantId);
                const metadata = message.metadata || {};
                (metadata.tool_calls || []).forEach((toolCall, index) => {
                    renderToolCallCard(assistantId, index, toolCall);
                });
                updateAssistantText(assistantId, message.content || '');
                renderMetadataCard(assistantId, metadata);
            }
        }
        await refreshConversationList();
        scrollToBottom();
    } catch (error) {
        console.error('Could not open conversation:', error);
    }
}

function startNewConversation() {
    if (isWorking) return;
    currentConversationId = null;
    localStorage.removeItem('localqwen-current-conversation');
    conversationTitle.textContent = 'New chat';
    messagesContainer.replaceChildren();
    renderConversationListFromCurrent();
    messageInput.focus();
}

async function renderConversationListFromCurrent() {
    try {
        await refreshConversationList();
    } catch (error) {
        console.error('Could not refresh conversation history:', error);
    }
}

function addUserMessage(content, scroll = true) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message user';
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '👤';
    const text = document.createElement('div');
    text.className = 'message-content user-message-text';
    text.textContent = content;
    wrapper.append(avatar, text);
    messagesContainer.appendChild(wrapper);
    if (scroll) scrollToBottom();
}

function renderAssistantShell(assistantId) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant';
    wrapper.id = `message-${assistantId}`;
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    const content = document.createElement('div');
    content.className = 'message-content';
    content.id = `message-content-${assistantId}`;
    wrapper.append(avatar, content);
    messagesContainer.appendChild(wrapper);
}

function updateAssistantText(assistantId, value) {
    const content = document.getElementById(`message-content-${assistantId}`);
    if (!content) return;
    let answer = document.getElementById(`answer-${assistantId}`);
    if (!answer) {
        answer = document.createElement('div');
        answer.className = 'assistant-text';
        answer.id = `answer-${assistantId}`;
        content.appendChild(answer);
    }
    answer.textContent = value;
    scrollToBottom();
}

function renderToolCallCard(assistantId, index, toolCall) {
    const content = document.getElementById(`message-content-${assistantId}`);
    if (!content) return;
    const card = document.createElement('details');
    card.className = 'tool-card';
    const summary = document.createElement('summary');
    summary.textContent = `🧮 Calculator · ${toolCall.success === false ? 'error' : `${toolCall.elapsed_ms ?? 0} ms`}`;
    card.appendChild(summary);
    appendMetadataLine(card, 'Expression', toolCall.expression || '');
    appendMetadataLine(card, toolCall.success === false ? 'Error' : 'Result', toolCall.error || toolCall.result || '');
    content.appendChild(card);
}

function renderMetadataCard(assistantId, metadata) {
    const content = document.getElementById(`message-content-${assistantId}`);
    if (!content || !metadata.timestamp_utc) return;
    const details = document.createElement('details');
    details.className = 'turn-metadata';
    const summary = document.createElement('summary');
    summary.textContent = 'Turn details';
    details.appendChild(summary);

    appendMetadataLine(details, 'Timestamp', formatDate(metadata.timestamp_utc, true));
    appendMetadataLine(details, 'Duration', `${metadata.duration_seconds ?? 0} s`);
    appendMetadataLine(details, 'Model', metadata.model_name || '');
    appendMetadataLine(details, 'Calculator used', metadata.tool_used ? 'Yes' : 'No');
    appendMetadataLine(details, 'Context / new token limit', `${metadata.max_length} / ${metadata.max_new_tokens}`);
    appendMetadataLine(details, 'Best-of-N', metadata.best_of_n);
    if (metadata.generation) {
        appendMetadataLine(
            details,
            'Sampling',
            `temp ${metadata.generation.temperature}; calibration ${metadata.generation.calibration_temperature}; top-k ${metadata.generation.top_k}; top-p ${metadata.generation.top_p}`
        );
    }
    if (metadata.accepted_candidate) {
        const accepted = metadata.accepted_candidate;
        appendMetadataLine(
            details,
            'Accepted candidate',
            `#${accepted.index} (${accepted.stage}; score ${formatScore(accepted.score, 4)})`
        );
    }
    (metadata.candidate_runs || []).forEach((run) => {
        const scores = (run.candidates || [])
            .map((candidate) => `${candidate.selected ? '*' : ''}#${candidate.index} ${formatScore(candidate.score, 3)}`)
            .join(' · ');
        appendMetadataLine(
            details,
            `Candidates · ${run.stage}`,
            `${run.prompt_tokens} prompt tokens; accepted #${run.accepted_candidate} / ${run.candidate_count}: ${scores}`
        );
    });
    content.appendChild(details);
}

function formatScore(value, digits) {
    return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function appendMetadataLine(parent, label, value) {
    const row = document.createElement('div');
    row.className = 'metadata-line';
    const labelElement = document.createElement('span');
    labelElement.className = 'metadata-label';
    labelElement.textContent = label;
    const valueElement = document.createElement('span');
    valueElement.className = 'metadata-value';
    valueElement.textContent = value ?? '';
    row.append(labelElement, valueElement);
    parent.appendChild(row);
}

function addLoadingIndicator(assistantId) {
    const content = document.getElementById(`message-content-${assistantId}`);
    const loading = document.createElement('div');
    loading.className = 'loading';
    loading.id = `loading-${assistantId}`;
    loading.innerHTML = '<div class="spinner"></div><span>Generating candidates and checking calculations…</span>';
    content.appendChild(loading);
}

function removeLoadingIndicator(assistantId) {
    document.getElementById(`loading-${assistantId}`)?.remove();
}

function formatDate(value, includeTime = false) {
    if (!value) return '';
    const date = new Date(value);
    return new Intl.DateTimeFormat(undefined, includeTime
        ? { dateStyle: 'medium', timeStyle: 'short' }
        : { dateStyle: 'medium' }).format(date);
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

initializeHistory();
