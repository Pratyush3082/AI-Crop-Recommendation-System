# recommender/views.py
import os
import joblib
import numpy as np
from django.views.decorators.http import require_http_methods
from django.contrib.auth import logout
from django.contrib import messages, auth
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login as auth_login, logout as auth_logout, authenticate
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import redirect
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from .utils import predict_crop, predict_crop_topk
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import datetime
import pandas as pd


# Lazy model loading: load on first use and cache
_MODEL_ARTIFACT = None
_MODEL_LOAD_ERROR = None

# Developer / contact info to include inside the PDF
DEVELOPER_NAME = "Pratyush Singh"
DEVELOPER_INSTITUTION = "IIMT College of Engineering, Greater Noida"
DEVELOPER_EMAIL = "prasingh3082@gmail.com"


def _load_model():
    """
    Load joblib artifact once and cache. Returns artifact dict or None (on error).
    """
    global _MODEL_ARTIFACT, _MODEL_LOAD_ERROR
    if _MODEL_ARTIFACT is not None or _MODEL_LOAD_ERROR is not None:
        return _MODEL_ARTIFACT
    try:
        model_path = os.path.join(settings.BASE_DIR, "recommender", "ml", "crop_recommender_v1.joblib")
        _MODEL_ARTIFACT = joblib.load(model_path)
        return _MODEL_ARTIFACT
    except Exception as e:
        _MODEL_LOAD_ERROR = e
        return None

@login_required(login_url='login')
def predict_view(request):
    """
    Renders the form and handles predictions. If the POST contains 'download',
    it generates a PDF which includes input values, prediction and developer details.
    """
    context = {}

    # Try to load model once to show any load errors to the template
    artifact_preview = _load_model()
    if artifact_preview is None and _MODEL_LOAD_ERROR is not None:
        context["load_error"] = str(_MODEL_LOAD_ERROR)

    if request.method == "POST":
        # Keep raw strings for re-display and for embedding into PDF hidden fields
        raw_data = {
            "N": request.POST.get("N", "").strip(),
            "P": request.POST.get("P", "").strip(),
            "K": request.POST.get("K", "").strip(),
            "temperature": request.POST.get("temperature", "").strip(),
            "humidity": request.POST.get("humidity", "").strip(),
            "ph": request.POST.get("ph", "").strip(),
            "rainfall": request.POST.get("rainfall", "").strip(),
        }

        # Parse numeric values; show friendly error if invalid
        try:
            data = {
                "N": float(raw_data["N"]),
                "P": float(raw_data["P"]),
                "K": float(raw_data["K"]),
                "temperature": float(raw_data["temperature"]),
                "humidity": float(raw_data["humidity"]),
                "ph": float(raw_data["ph"]),
                "rainfall": float(raw_data["rainfall"]),
            }
        except ValueError:
            context["error"] = "All input fields must be valid numbers."
            context["data"] = raw_data
            return render(request, "recommender/predict.html", context)

        # --- Normalize top3 here so template gets stable format: list of dicts
        # each item: {"crop": "Rice", "confidence": 87.23} where confidence is percent (0..100)
        normalized_top3 = []
        try:
            _raw_top3 = predict_crop_topk(data, k=3)
            if isinstance(_raw_top3, (list, tuple)):
                for it in _raw_top3:
                    if isinstance(it, dict):
                        crop_name = it.get("crop") or it.get("label") or it.get("name") or ""
                        conf = it.get("confidence", it.get("prob", 0))
                    elif isinstance(it, (list, tuple)) and len(it) >= 2:
                        crop_name, conf = it[0], it[1]
                    else:
                        continue
                    # coerce conf to float, convert fraction->percent if needed
                    try:
                        conf_f = float(conf)
                        if 0.0 <= conf_f <= 1.0:
                            conf_pct = conf_f * 100.0
                        else:
                            conf_pct = conf_f
                    except Exception:
                        conf_pct = 0.0
                    normalized_top3.append({"crop": str(crop_name), "confidence": round(conf_pct, 2)})
        except Exception:
            normalized_top3 = []

        context["top3"] = normalized_top3

        # Ensure model is loaded
        artifact = _load_model()
        if artifact is None:
            context["error"] = f"Model load failed: {_MODEL_LOAD_ERROR}"
            context["data"] = raw_data
            return render(request, "recommender/predict.html", context)

        # Run prediction using your utility function
        try:
            res = predict_crop(data)
            # support both (crop, conf, probs_map) and (crop, conf)
            if isinstance(res, tuple) and len(res) == 3:
                crop, confidence, probs_map = res
            elif isinstance(res, tuple) and len(res) == 2:
                crop, confidence = res
                probs_map = None
            else:
                raise RuntimeError("predict_crop returned unexpected format")

            # normalize main confidence to percent (0..100) for template
            try:
                conf_f = float(confidence)
                if 0.0 <= conf_f <= 1.0:
                    conf_pct = conf_f * 100.0
                else:
                    conf_pct = conf_f
            except Exception:
                # fallback: 0..100 expected
                conf_pct = float(confidence) if confidence is not None else 0.0

            context["result"] = {"crop": crop, "confidence": round(conf_pct, 2)}
            context["data"] = raw_data
            if probs_map is not None:
                # probs_map expected as fractions 0..1 — keep as-is (used in PDF as fractions)
                context["result"]["probs"] = probs_map
        except Exception as e:
            context["error"] = f"Prediction failed: {e}"
            context["data"] = raw_data
            return render(request, "recommender/predict.html", context)

        # If user requested download, generate PDF (with developer/contact details)
        if "download" in request.POST:
            try:
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=A4)
                width, height = A4
                p.setTitle("Crop Recommendation Report")

                # --- Header: Title centered, date top-right ---
                p.setFont("Helvetica-Bold", 16)
                title_x = width / 2
                p.drawCentredString(title_x, height - 50, "AI Crop Recommendation Report")

                p.setFont("Helvetica", 9)
                date_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                p.drawRightString(width - 40, height - 45, f"Date: {date_text}")

                # Draw a thin line under header
                p.setLineWidth(0.5)
                p.line(40, height - 60, width - 40, height - 60)

                # --- Inputs ---
                p.setFont("Helvetica-Bold", 12)
                y = height - 90
                p.drawString(50, y, "Input Parameters:")
                p.setFont("Helvetica", 11)
                y -= 18
                for key, value in raw_data.items():
                    p.drawString(70, y, f"{key.capitalize()}: {value}")
                    y -= 16

                # --- Prediction ---
                y -= 6
                p.setFont("Helvetica-Bold", 13)
                p.drawString(50, y, f"Predicted Crop: {str(context['result']['crop']).title()}")
                y -= 18
                p.setFont("Helvetica", 11)
                # context["result"]["confidence"] is percent (0..100)
                p.drawString(50, y, f"Confidence: {context['result']['confidence']:.2f}%")

                # --- Top probabilities (if available) ---
                if 'probs' in context["result"] and context["result"]["probs"]:
                    probs_map = context["result"]["probs"]
                    y -= 24
                    p.setFont("Helvetica-Bold", 12)
                    p.drawString(50, y, "Top probabilities:")
                    y -= 18
                    # sort and take top 5
                    sorted_items = sorted(probs_map.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    p.setFont("Helvetica", 11)
                    for name, prob in sorted_items:
                        # prob assumed fraction -> convert to percent
                        try:
                            p.drawString(70, y, f"{str(name).title()}: {float(prob) * 100:.2f}%")
                        except Exception:
                            p.drawString(70, y, f"{str(name).title()}: {prob}")
                        y -= 16

                # If probs_map not provided by predict_crop, try to compute from artifact (best-effort)
                elif artifact is not None:
                    try:
                        feat_order = artifact.get("features")
                        model = artifact.get("model")
                        le = artifact.get("label_encoder")
                        if feat_order and model is not None and le is not None:
                            x = np.array([[float(raw_data.get(f) or 0) for f in feat_order]])
                            probs = model.predict_proba(x)[0]
                            classes = list(le.classes_)
                            class_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:5]
                            y -= 24
                            p.setFont("Helvetica-Bold", 12)
                            p.drawString(50, y, "Top probabilities:")
                            y -= 18
                            p.setFont("Helvetica", 11)
                            for cname, pval in class_probs:
                                p.drawString(70, y, f"{cname.title()}: {pval * 100:.2f}%")
                                y -= 16
                    except Exception:
                        # silently skip if computing probs fails
                        pass

                # --- Developer / Contact details (inside the PDF) ---
                footer_y = 140
                p.setFont("Helvetica-Bold", 12)
                p.drawString(50, footer_y + 40, "Contact / Developer Details")
                p.setFont("Helvetica", 11)
                p.drawString(60, footer_y + 20, f"Developed by: {DEVELOPER_NAME}")
                p.drawString(60, footer_y + 6, f"Institution: {DEVELOPER_INSTITUTION}")
                p.drawString(60, footer_y - 8, f"Email: {DEVELOPER_EMAIL}")

                # --- Footer small text ---
                p.setFont("Helvetica-Oblique", 9)
                p.drawString(50, footer_y - 36, "Generated by AI Crop Recommendation System")
                p.drawRightString(width - 40, footer_y - 36, f"Report generated: {date_text}")

                # finalize PDF
                p.showPage()
                p.save()
                buffer.seek(0)
                # send the PDF as a downloadable file
                response = HttpResponse(buffer, content_type="application/pdf")
                response["Content-Disposition"] = 'attachment; filename="Crop_Recommendation_Report.pdf"'
                return response

            except Exception as e:
                context["error"] = f"Failed to generate PDF: {e}"
                context["data"] = raw_data
                return render(request, "recommender/predict.html", context)

    # GET or fallback render
    return render(request, "recommender/predict.html", context)


def predict_api(request):
    """
    POST JSON API:
    { "N":.., "P":.., "K":.., "temperature":.., "humidity":.., "ph":.., "rainfall":.. }
    Returns: {"crop": "...", "confidence": 0.1234, "top3": [{"crop": "...", "confidence": 87.23}, ...]}
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST JSON only"}, status=400)

    try:
        import json
        payload = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # parse numeric fields with defaults
    try:
        data = {
            "N": float(payload.get("N", 0)),
            "P": float(payload.get("P", 0)),
            "K": float(payload.get("K", 0)),
            "temperature": float(payload.get("temperature", 0)),
            "humidity": float(payload.get("humidity", 0)),
            "ph": float(payload.get("ph", 0)),
            "rainfall": float(payload.get("rainfall", 0)),
        }
    except (ValueError, TypeError):
        return JsonResponse({"error": "Numeric fields must be numbers"}, status=400)

    # Ensure model is available
    artifact = _load_model()
    if artifact is None:
        return JsonResponse({"error": f"Model load failed: {_MODEL_LOAD_ERROR}"}, status=500)

    try:
        # compute top3 (for API consumers)
        normalized_top3 = []
        try:
            _raw_top3 = predict_crop_topk(data, k=3)
            if isinstance(_raw_top3, (list, tuple)):
                for it in _raw_top3:
                    if isinstance(it, dict):
                        crop_name = it.get("crop") or it.get("label") or it.get("name") or ""
                        conf = it.get("confidence", it.get("prob", 0))
                    elif isinstance(it, (list, tuple)) and len(it) >= 2:
                        crop_name, conf = it[0], it[1]
                    else:
                        continue
                    try:
                        conf_f = float(conf)
                        if 0.0 <= conf_f <= 1.0:
                            conf_pct = conf_f * 100.0
                        else:
                            conf_pct = conf_f
                    except Exception:
                        conf_pct = 0.0
                    normalized_top3.append({"crop": str(crop_name), "confidence": round(conf_pct, 2)})
        except Exception:
            normalized_top3 = []

        res = predict_crop(data)
        if isinstance(res, tuple) and len(res) == 3:
            crop, confidence, probs_map = res
        elif isinstance(res, tuple) and len(res) == 2:
            crop, confidence = res
            probs_map = None
        else:
            raise RuntimeError("predict_crop returned unexpected result")

        # convert confidence to float (API returns fraction 0..1 for model confidence)
        conf_float = float(confidence)
        # return API confidence as fraction (0..1) to be consistent with model / consumers
        response_data = {
            "crop": crop,
            "confidence": round(conf_float, 4),
            "top3": normalized_top3,  # percent-format for convenience
        }
        if probs_map is not None:
            try:
                response_data["probs"] = {str(k): float(v) for k, v in probs_map.items()}
            except Exception:
                pass

        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({"error": f"Prediction failed: {e}"}, status=500)
@staff_member_required
def retrain_view(request):
    if request.method == "POST" and request.FILES.get("csv"):
        f = request.FILES["csv"]
        df = pd.read_csv(f)
        df.to_csv(os.path.join(settings.BASE_DIR, "data", "crop_data.csv"), index=False)
        # optionally call the training script programmatically
        import subprocess
        subprocess.run([os.path.join(settings.BASE_DIR,".venv","Scripts","python.exe"), os.path.join(settings.BASE_DIR,"scripts","train_balanced.py")])
        return redirect("admin:index")
    return render(request, "recommender/retrain.html")
def signup_view(request):
    if request.user.is_authenticated:
        return redirect("predict")
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            u = form.save()
            # optional - log the user in immediately
            auth_login(request, u)
            messages.success(request, "Account created — welcome!")
            return redirect("predict")
        else:
            messages.error(request, "Please fix the errors below.")
    else:
        form = UserCreationForm()
    return render(request, "recommender/signup.html", {"form": form})
                  
# ----- Login -----
def login_view(request):
    if request.user.is_authenticated:
        return redirect("predict")  # already logged in
    form = AuthenticationForm(request=request, data=request.POST or None)
    if request.method == "POST":
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            messages.success(request, "Welcome back!")
            next_url = request.GET.get("next") or reverse("predict")
            return redirect(next_url)
        else:
            messages.error(request, "Invalid credentials. Please try again.")
    return render(request, "recommender/login.html", {"form": form})
@require_http_methods(["GET", "POST"])
def logout_view(request):
    logout(request)
    return redirect("login")

