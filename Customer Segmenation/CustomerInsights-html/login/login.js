function validate() {
    var email = document.getElementById("email").value;
    var password = document.getElementById("password").value;

    if (email == null || email == "") {
        alert("Email can't be blank");
        return false;
    }
    else if (password.length < 6) {
        alert("Password must be at least 6 characters long.");
        return false;
    }
    else if (password.length >= 20) {
        alert("Password must be less than 20  characters long.");
        return false;
    }

    else {
        alert("login succesfully");
        return false;

    }
}
