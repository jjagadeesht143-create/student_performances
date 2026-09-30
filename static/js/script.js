// Auto-dismiss alerts after 5 seconds
document.addEventListener("DOMContentLoaded", function () {
    const alerts = document.querySelectorAll(".alert");
    alerts.forEach(function (alert) {
        setTimeout(function () {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        }, 6000);
    });

    const toggle = document.querySelector("[data-password-toggle]");
    const password = document.getElementById("password");
    if (toggle && password) {
        toggle.addEventListener("click", function () {
            const visible = password.type === "text";
            password.type = visible ? "password" : "text";
            toggle.setAttribute("aria-label", visible ? "Show password" : "Hide password");
            toggle.innerHTML = `<i class="fa-solid fa-eye${visible ? "" : "-slash"}"></i>`;
        });
    }

    const loginForm = document.getElementById("login-form");
    if (loginForm) {
        loginForm.addEventListener("submit", function () {
            const button = loginForm.querySelector(".login-submit");
            if (button) {
                button.disabled = true;
                button.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Signing in';
            }
        });
    }

    document.querySelectorAll("[data-google-login]").forEach(function (button) {
        button.addEventListener("click", function () {
            button.classList.add("disabled");
            button.setAttribute("aria-disabled", "true");
            button.innerHTML = '<i class="fa-solid fa-spinner fa-spin" aria-hidden="true"></i><span>Connecting to Google...</span>';
        });
    });

    const predictionForm = document.querySelector("[data-prediction-form]");
    if (predictionForm) {
        predictionForm.addEventListener("submit", function (event) {
            if (!predictionForm.checkValidity()) return;
            const button = predictionForm.querySelector(".prediction-submit");
            if (button) {
                button.disabled = true;
                button.querySelector("span").textContent = "Generating prediction...";
                button.querySelector("i").className = "fa-solid fa-spinner fa-spin me-2";
            }
        });
    }

    const predictionReset = document.querySelector("[data-prediction-reset]");
    if (predictionReset && predictionForm) {
        predictionReset.addEventListener("click", function () {
            const resultCard = document.querySelector(".prediction-result-card");
            if (resultCard) resultCard.remove();
            window.setTimeout(function () {
                predictionForm.querySelectorAll("input").forEach(function (field) { field.value = ""; });
                predictionForm.querySelectorAll("select").forEach(function (field) { field.selectedIndex = 0; });
                predictionForm.querySelectorAll(".is-invalid").forEach(function (field) { field.classList.remove("is-invalid"); });
            }, 0);
        });
    }
});
