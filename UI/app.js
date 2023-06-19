function listAccessibleCustomers() {
    const url = 'http://127.0.0.1:5000/api/interpret';
    const xhr = XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send();
    xhr.onload = function() {
        const response = JSON.parse(xhr.responseText);
        console.log(response);
    }
}