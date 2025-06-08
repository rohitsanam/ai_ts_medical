// Global variables
let selectedFile = null;
let uploadedImageData = null;
let currentLanguage = 'english';
let analysisResults = null; // Store complete analysis results for report generation
let isAnalyzing = false; // Flag to prevent multiple analysis runs

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Translation system
const translations = {
    english: {
        // Page 1 - Welcome
        welcomeTitle: "MEDICAL IMAGE AI",
        welcomeSubtitle: "Advanced AI-powered medical image classification system",
        startAnalysis: "START ANALYSIS",
        
        // Page 2 - Language Selection
        selectLanguageTitle: "Select Language",
        selectLanguageSubtitle: "Choose your preferred language to continue",
        selectLanguageLabel: "Select Language:",
        chooseLanguage: "-- Choose Language --",
        backBtn: "BACK",
        initializeBtn: "INITIALIZE",
        
        // Page 3 - Upload
        uploadTitle: "MEDICAL IMAGE ANALYSIS",
        uploadSubtitle: "Upload your DICOM file or medical image for AI-powered analysis",
        uploadSectionTitle: "UPLOAD MEDICAL IMAGE",
        uploadText: "Drag & drop your DICOM file or medical image here",
        uploadFormats: "Supports: DICOM (.dcm), PNG, JPG, JPEG",
        browseFiles: "Browse files",
        analyzeBtn: "ANALYZE",
        
        // Page 4 - Processing
        processingTitle: "PROCESSING",
        processingSubtitle: "AI analysis in progress...",
        analyzingImage: "ANALYZING MEDICAL IMAGE",
        pleaseWait: "Please wait while our AI processes your image",
        
        // Page 5 - Results
        resultsTitle: "ANALYSIS RESULTS",
        resultsSubtitle: "Comprehensive AI-powered medical image analysis",
        uploadedImage: "UPLOADED IMAGE",
        diagnosisResults: "DIAGNOSIS RESULTS",
        primaryDiagnosis: "Primary Diagnosis:",
        topClassification: "TOP CLASSIFICATION",
        patientInformation: "PATIENT INFORMATION",
        downloadReport: "DOWNLOAD REPORT",
        newAnalysis: "NEW ANALYSIS",
        
        // Patient Info Labels
        patientId: "Patient ID:",
        patientName: "Patient Name:",
        sex: "Sex:",
        age: "Age:",
        weight: "Weight:",
        
        // Footer
        importantNote: "Important: This AI system is designed to assist medical professionals and should not be used as the sole basis for medical diagnosis. Always consult qualified healthcare providers for proper medical interpretation and decision-making.",
        secureProcessing: "Secure, HIPAA-compliant processing • All data is encrypted"
    },
    
    telugu: {
        // Page 1 - Welcome
        welcomeTitle: "వైద్య చిత్ర AI",
        welcomeSubtitle: "అధునాతన AI-శక్తితో వైద్య చిత్ర వర్గీకరణ వ్యవస్థ",
        startAnalysis: "విశ్లేషణ ప్రారంభించండి",
        
        // Page 2 - Language Selection
        selectLanguageTitle: "భాష ఎంచుకోండి",
        selectLanguageSubtitle: "కొనసాగించడానికి మీ ఇష్టపడే భాషను ఎంచుకోండి",
        selectLanguageLabel: "భాష ఎంచుకోండి:",
        chooseLanguage: "-- భాష ఎంచుకోండి --",
        backBtn: "వెనుకకు",
        initializeBtn: "ప్రారంభించండి",
        
        // Page 3 - Upload
        uploadTitle: "వైద్య చిత్ర విశ్లేషణ",
        uploadSubtitle: "AI-శక్తితో విశ్లేషణ కోసం మీ DICOM ఫైల్ లేదా వైద్య చిత్రాన్ని అప్‌లోడ్ చేయండి",
        uploadSectionTitle: "వైద్య చిత్రం అప్‌లోడ్ చేయండి",
        uploadText: "మీ DICOM ఫైల్ లేదా వైద్య చిత్రాన్ని ఇక్కడ లాగి వదలండి",
        uploadFormats: "మద్దతు: DICOM (.dcm), PNG, JPG, JPEG",
        browseFiles: "ఫైల్‌లను బ్రౌజ్ చేయండి",
        analyzeBtn: "విశ్లేషించండి",
        
        // Page 4 - Processing
        processingTitle: "ప్రాసెసింగ్",
        processingSubtitle: "AI విశ్లేషణ ప్రోగ్రెస్‌లో...",
        analyzingImage: "వైద్య చిత్రాన్ని విశ్లేషిస్తోంది",
        pleaseWait: "మా AI మీ చిత్రాన్ని ప్రాసెస్ చేస్తున్నప్పుడు దయచేసి వేచి ఉండండి",
        
        // Page 5 - Results
        resultsTitle: "విశ్లేషణ ఫలితాలు",
        resultsSubtitle: "సమగ్ర AI-శక్తితో వైద్య చిత్ర విశ్లేషణ",
        uploadedImage: "అప్‌లోడ్ చేసిన చిత్రం",
        diagnosisResults: "రోగనిర్ధారణ ఫలితాలు",
        primaryDiagnosis: "ప్రాథమిక రోగనిర్ధారణ:",
        topClassification: "టాప్ వర్గీకరణ",
        patientInformation: "రోగి సమాచారం",
        downloadReport: "రిపోర్ట్ డౌన్‌లోడ్ చేయండి",
        newAnalysis: "కొత్త విశ్లేషణ",
        
        // Patient Info Labels
        patientId: "రోగి ID:",
        patientName: "రోగి పేరు:",
        sex: "లింగం:",
        age: "వయస్సు:",
        weight: "బరువు:",
        
        // Footer
        importantNote: "ముఖ్యమైనది: ఈ AI వ్యవస్థ వైద్య నిపుణులకు సహాయం చేయడానికి రూపొందించబడింది మరియు వైద్య రోగనిర్ధారణకు ఏకైక ఆధారంగా ఉపయోగించరాదు. సరైన వైద్య వివరణ మరియు నిర్ణయం తీసుకోవడానికి ఎల్లప్పుడూ అర్హతైన ఆరోగ్య సేవా ప్రదాతలను సంప్రదించండి.",
        secureProcessing: "భద్రతైన, HIPAA-అనుకూల ప్రాసెసింగ్ • అన్ని డేటా ఎన్‌క్రిప్ట్ చేయబడింది"
    },
    
    hindi: {
        // Page 1 - Welcome
        welcomeTitle: "मेडिकल इमेज AI",
        welcomeSubtitle: "उन्नत AI-संचालित चिकित्सा छवि वर्गीकरण प्रणाली",
        startAnalysis: "विश्लेषण शुरू करें",
        
        // Page 2 - Language Selection
        selectLanguageTitle: "भाषा चुनें",
        selectLanguageSubtitle: "जारी रखने के लिए अपनी पसंदीदा भाषा चुनें",
        selectLanguageLabel: "भाषा चुनें:",
        chooseLanguage: "-- भाषा चुनें --",
        backBtn: "वापस",
        initializeBtn: "प्रारंभ करें",
        
        // Page 3 - Upload
        uploadTitle: "चिकित्सा छवि विश्लेषण",
        uploadSubtitle: "AI-संचालित विश्लेषण के लिए अपनी DICOM फ़ाइल या चिकित्सा छवि अपलोड करें",
        uploadSectionTitle: "चिकित्सा छवि अपलोड करें",
        uploadText: "अपनी DICOM फ़ाइल या चिकित्सा छवि को यहाँ खींचें और छोड़ें",
        uploadFormats: "समर्थित: DICOM (.dcm), PNG, JPG, JPEG",
        browseFiles: "फ़ाइलें ब्राउज़ करें",
        analyzeBtn: "विश्लेषण करें",
        
        // Page 4 - Processing
        processingTitle: "प्रसंस्करण",
        processingSubtitle: "AI विश्लेषण प्रगति में...",
        analyzingImage: "चिकित्सा छवि का विश्लेषण",
        pleaseWait: "कृपया प्रतीक्षा करें जबकि हमारा AI आपकी छवि को प्रोसेस करता है",
        
        // Page 5 - Results
        resultsTitle: "विश्लेषण परिणाम",
        resultsSubtitle: "व्यापक AI-संचालित चिकित्सा छवि विश्लेषण",
        uploadedImage: "अपलोड की गई छवि",
        diagnosisResults: "निदान परिणाम",
        primaryDiagnosis: "प्राथमिक निदान:",
        topClassification: "शीर्ष वर्गीकरण",
        patientInformation: "रोगी की जानकारी",
        downloadReport: "रिपोर्ट डाउनलोड करें",
        newAnalysis: "नया विश्लेषण",
        
        // Patient Info Labels
        patientId: "रोगी ID:",
        patientName: "रोगी का नाम:",
        sex: "लिंग:",
        age: "आयु:",
        weight: "वजन:",
        
        // Footer
        importantNote: "महत्वपूर्ण: यह AI प्रणाली चिकित्सा पेशेवरों की सहायता के लिए डिज़ाइन की गई है और इसका उपयोग चिकित्सा निदान के एकमात्र आधार के रूप में नहीं किया जाना चाहिए। उचित चिकित्सा व्याख्या और निर्णय लेने के लिए हमेशा योग्य स्वास्थ्य सेवा प्रदाताओं से सलाह लें।",
        secureProcessing: "सुरक्षित, HIPAA-अनुपालित प्रसंस्करण • सभी डेटा एन्क्रिप्ट किया गया है"
    },
    
    urdu: {
        // Page 1 - Welcome
        welcomeTitle: "طبی تصویر AI",
        welcomeSubtitle: "جدید AI پر مبنی طبی تصویر کی درجہ بندی کا نظام",
        startAnalysis: "تجزیہ شروع کریں",
        
        // Page 2 - Language Selection
        selectLanguageTitle: "زبان منتخب کریں",
        selectLanguageSubtitle: "جاری رکھنے کے لیے اپنی پسندیدہ زبان منتخب کریں",
        selectLanguageLabel: "زبان منتخب کریں:",
        chooseLanguage: "-- زبان منتخب کریں --",
        backBtn: "واپس",
        initializeBtn: "شروع کریں",
        
        // Page 3 - Upload
        uploadTitle: "طبی تصویر کا تجزیہ",
        uploadSubtitle: "AI پر مبنی تجزیے کے لیے اپنی DICOM فائل یا طبی تصویر اپ لوڈ کریں",
        uploadSectionTitle: "طبی تصویر اپ لوڈ کریں",
        uploadText: "اپنی DICOM فائل یا طبی تصویر یہاں کھینچ کر چھوڑیں",
        uploadFormats: "تعاون: DICOM (.dcm), PNG, JPG, JPEG",
        browseFiles: "فائلز براؤز کریں",
        analyzeBtn: "تجزیہ کریں",
        
        // Page 4 - Processing
        processingTitle: "پروسیسنگ",
        processingSubtitle: "AI تجزیہ جاری ہے...",
        analyzingImage: "طبی تصویر کا تجزیہ",
        pleaseWait: "برائے کرم انتظار کریں جبکہ ہمارا AI آپ کی تصویر پر کام کرتا ہے",
        
        // Page 5 - Results
        resultsTitle: "تجزیے کے نتائج",
        resultsSubtitle: "جامع AI پر مبنی طبی تصویر کا تجزیہ",
        uploadedImage: "اپ لوڈ شدہ تصویر",
        diagnosisResults: "تشخیص کے نتائج",
        primaryDiagnosis: "بنیادی تشخیص:",
        topClassification: "اعلیٰ درجہ بندی",
        patientInformation: "مریض کی معلومات",
        downloadReport: "رپورٹ ڈاؤن لوڈ کریں",
        newAnalysis: "نیا تجزیہ",
        
        // Patient Info Labels
        patientId: "مریض کی ID:",
        patientName: "مریض کا نام:",
        sex: "جنس:",
        age: "عمر:",
        weight: "وزن:",
        
        // Footer
        importantNote: "اہم: یہ AI سسٹم طبی پیشہ ور افراد کی مدد کے لیے بنایا گیا ہے اور اسے طبی تشخیص کی واحد بنیاد کے طور پر استعمال نہیں کرنا چاہیے۔ مناسب طبی تشریح اور فیصلہ سازی کے لیے ہمیشہ قابل صحت کی دیکھ بھال فراہم کرنے والوں سے مشورہ کریں۔",
        secureProcessing: "محفوظ، HIPAA کے مطابق پروسیسنگ • تمام ڈیٹا خفیہ کار ہے"
    }
};

// Translation function
function t(key) {
    return translations[currentLanguage] && translations[currentLanguage][key] 
        ? translations[currentLanguage][key] 
        : translations.english[key] || key;
}

// Function to update all text on the page
function updateLanguage() {
    console.log(`🌐 Updating language to: ${currentLanguage}`);
    
    // Only update elements that exist to avoid errors
    try {
        // Page 1 - Welcome
        const welcomeTitle = document.querySelector('#page1 h1');
        const welcomeSubtitle = document.querySelector('#page1 p');
        const startAnalysisBtn = document.querySelector('#next-to-page2');
        
        if (welcomeTitle) welcomeTitle.textContent = t('welcomeTitle');
        if (welcomeSubtitle) welcomeSubtitle.textContent = t('welcomeSubtitle');
        if (startAnalysisBtn) {
            // DON'T change innerHTML - just update the text content to preserve event listeners
            const svgElement = startAnalysisBtn.querySelector('svg');
            startAnalysisBtn.textContent = t('startAnalysis') + ' ';
            if (svgElement) {
                startAnalysisBtn.appendChild(svgElement);
            }
        }
        
        // Page 2 - Language Selection
        const langTitle = document.querySelector('#page2 h2');
        const langSubtitle = document.querySelector('#page2 p');
        const langLabel = document.querySelector('label[for="languageSelect"]');
        const langOption = document.querySelector('#languageSelect option[value=""]');
        const backToPage1 = document.querySelector('#back-to-page1');
        const initializeBtn = document.querySelector('#nextBtn');
        
        if (langTitle) langTitle.textContent = t('selectLanguageTitle');
        if (langSubtitle) langSubtitle.textContent = t('selectLanguageSubtitle');
        if (langLabel) langLabel.textContent = t('selectLanguageLabel');
        if (langOption) langOption.textContent = t('chooseLanguage');
        if (backToPage1) {
            const svgElement = backToPage1.querySelector('svg');
            backToPage1.textContent = ' ' + t('backBtn');
            if (svgElement) {
                backToPage1.insertBefore(svgElement, backToPage1.firstChild);
            }
        }
        if (initializeBtn) initializeBtn.textContent = t('initializeBtn');
        
        // Page 3 - Upload
        const uploadTitle = document.querySelector('#page3 h1');
        const uploadSubtitle = document.querySelector('#page3 > div > div:first-child p');
        const uploadSectionTitle = document.querySelector('#page3 h2');
        const uploadText = document.querySelector('#page3 .upload-area p:first-child');
        const uploadFormats = document.querySelector('#page3 .upload-area p:nth-child(2)');
        const browseFiles = document.querySelector('#page3 .upload-area label span');
        const backToPage2 = document.querySelector('#back-to-page2');
        const analyzeBtn = document.querySelector('#analyze-btn-text');
        
        if (uploadTitle) uploadTitle.textContent = t('uploadTitle');
        if (uploadSubtitle) uploadSubtitle.textContent = t('uploadSubtitle');
        if (uploadSectionTitle) uploadSectionTitle.textContent = t('uploadSectionTitle');
        if (uploadText) uploadText.textContent = t('uploadText');
        if (uploadFormats) uploadFormats.textContent = t('uploadFormats');
        if (browseFiles) browseFiles.textContent = t('browseFiles');
        if (backToPage2) {
            const svgElement = backToPage2.querySelector('svg');
            backToPage2.textContent = ' ' + t('backBtn');
            if (svgElement) {
                backToPage2.insertBefore(svgElement, backToPage2.firstChild);
            }
        }
        if (analyzeBtn) analyzeBtn.textContent = t('analyzeBtn');
        
        // Page 4 - Processing
        const processingTitle = document.querySelector('#page4 h1');
        const processingSubtitle = document.querySelector('#page4 > div > div:first-child p');
        const analyzingImage = document.querySelector('#page4 h2');
        const pleaseWait = document.querySelector('#page4 .card p');
        
        if (processingTitle) processingTitle.textContent = t('processingTitle');
        if (processingSubtitle) processingSubtitle.textContent = t('processingSubtitle');
        if (analyzingImage) analyzingImage.textContent = t('analyzingImage');
        if (pleaseWait) pleaseWait.textContent = t('pleaseWait');
        
        // Page 5 - Results
        const resultsTitle = document.querySelector('#page5 h1');
        const resultsSubtitle = document.querySelector('#page5 > div > div:first-child p');
        const uploadedImageTitle = document.querySelector('#page5 .card h3:first-child');
        const diagnosisResultsTitle = document.querySelector('#page5 .card h3:nth-child(2)');
        const topClassTitle = document.querySelector('#page5 h4:first-child');
        const patientInfoTitle = document.querySelector('#page5 h4:nth-child(2)');
        const downloadReportBtn = document.querySelector('#download-report');
        const newAnalysisBtn = document.querySelector('#new-analysis');
        
        if (resultsTitle) resultsTitle.textContent = t('resultsTitle');
        if (resultsSubtitle) resultsSubtitle.textContent = t('resultsSubtitle');
        if (uploadedImageTitle) uploadedImageTitle.textContent = t('uploadedImage');
        if (diagnosisResultsTitle) diagnosisResultsTitle.textContent = t('diagnosisResults');
        if (topClassTitle) topClassTitle.textContent = t('topClassification');
        if (patientInfoTitle) patientInfoTitle.textContent = t('patientInformation');
        if (downloadReportBtn) downloadReportBtn.textContent = t('downloadReport');
        if (newAnalysisBtn) newAnalysisBtn.textContent = t('newAnalysis');
        
        // Footer messages - be more careful here too
        const footerMessages = document.querySelectorAll('#page3 .text-center p, #page4 .text-center p, #page5 .text-center p');
        footerMessages.forEach(msg => {
            if (msg.textContent.includes('Secure') || msg.textContent.includes('భద్రతైన') || msg.textContent.includes('सुरक्षित') || msg.textContent.includes('محفوظ')) {
                msg.textContent = t('secureProcessing');
            } else if (msg.textContent.includes('Important') || msg.textContent.includes('ముఖ్యమైనది') || msg.textContent.includes('महत्वपूर्ण') || msg.textContent.includes('اہم')) {
                msg.textContent = t('importantNote');
            }
        });
        
        console.log(`✅ Language updated to ${currentLanguage}`);
    } catch (error) {
        console.warn('⚠️ Some elements not found during language update:', error);
    }
}

// Initialize variables for DOM elements (will be set in DOMContentLoaded)
let pages = {};
let languageSelect, nextBtn;
let fileDropArea, fileInput, fileInfo, fileName, fileSize, removeFileBtn, errorMessage, analyzeBtn, analyzeBtnText;
let progressBar, progressText;
let resultImage, resultFilename, primaryDiagnosis, confidenceScore, allClassifications, patientInfo;
let downloadReportBtn, newAnalysisBtn;

// Page Navigation Functions
function showPage(pageId) {
    console.log(`🔄 Switching to page: ${pageId}`);
    
    // Hide all pages
    Object.values(pages).forEach(page => {
        if (page) {
            page.classList.remove('active');
            page.classList.add('inactive');
        }
    });
    
    // Show target page
    const targetPage = pages[pageId];
    if (targetPage) {
        targetPage.classList.remove('inactive');
        targetPage.classList.add('active');
        console.log(`✅ Page ${pageId} is now active`);
        console.log(`Page classes:`, targetPage.className);
    } else {
        console.error(`❌ Page ${pageId} not found`);
    }
}

// File Upload Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
    setTimeout(() => {
        errorMessage.classList.add('hidden');
    }, 5000);
}

function validateFile(file) {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const allowedTypes = [
        'application/dicom',
        'image/png', 
        'image/jpeg', 
        'image/jpg'
    ];
    const allowedExtensions = ['.dcm', '.png', '.jpg', '.jpeg'];
    
    if (file.size > maxSize) {
        showError('File size must be less than 50MB');
        return false;
    }
    
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
        showError('Please upload a DICOM file (.dcm) or image file (PNG, JPG, JPEG)');
        return false;
    }
    
    return true;
}

function handleFileSelection(file) {
    if (!validateFile(file)) {
        return;
    }
    
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    errorMessage.classList.add('hidden');
}

function resetFileSelection() {
    selectedFile = null;
    uploadedImageData = null;
    analysisResults = null;
    fileInfo.classList.add('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');
    errorMessage.classList.add('hidden');
    fileInput.value = '';
}

// Progress Functions
function updateProgress(percentage, message) {
    if (progressBar) {
        progressBar.style.width = percentage + '%';
    }
    if (progressText) {
        progressText.textContent = message;
    }
}

// Optimized API Functions - Single call for both upload and prediction
async function analyzeImageAPI(file) {
    try {
        updateProgress(20, 'Uploading and analyzing image...');
        
        // Single API call for complete analysis (replaces both upload and predict)
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/analyze/`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Analysis failed: ${response.status} ${response.statusText}`);
        }
        
        updateProgress(60, 'Processing analysis results...');
        
        const analysisData = await response.json();
        updateProgress(100, 'Analysis complete!');
        
        // Store complete analysis results with the processed image from backend
        analysisResults = {
            ...analysisData,
            imageData: analysisData.ProcessedImage, // Backend now returns processed image
            fileName: file.name,
            fileSize: formatFileSize(file.size),
            analysisDate: new Date().toISOString(),
            language: currentLanguage
        };
        
        return analysisResults;
        
    } catch (error) {
        throw new Error(`Analysis failed: ${error.message}`);
    }
}

// Results Display Functions
function displayResults(results) {
    // Display uploaded image
    if (resultImage && results.imageData) {
        resultImage.src = results.imageData;
        resultImage.alt = "Uploaded medical image";
    }
    
    if (resultFilename) {
        resultFilename.textContent = results.FileName || results.fileName || 'Unknown file';
    }
    
    // Display primary diagnosis
    if (primaryDiagnosis) {
        primaryDiagnosis.textContent = results.Diagnosis || 'Unknown';
        
        // Color coding based on diagnosis
        const diagnosis = results.Diagnosis;
        if (diagnosis === 'Normal') {
            primaryDiagnosis.className = 'text-2xl font-bold mb-2 tech-font text-green-400';
        } else if (diagnosis === 'TB Positive') {
            primaryDiagnosis.className = 'text-2xl font-bold mb-2 tech-font text-red-400';
        } else if (diagnosis === 'Lung Cancer') {
            primaryDiagnosis.className = 'text-2xl font-bold mb-2 tech-font text-red-500';
        } else {
            primaryDiagnosis.className = 'text-2xl font-bold mb-2 tech-font text-yellow-400';
        }
    }
    
    if (confidenceScore) {
        confidenceScore.textContent = `Confidence: ${results.Confidence}%`;
    }
    
    // Display only the highest classification probability (argmax)
    if (allClassifications && results.AllClassConfidences) {
        allClassifications.innerHTML = '';
        
        // Find the highest confidence classification
        let maxConfidence = 0;
        let topClassification = '';
        
        Object.entries(results.AllClassConfidences).forEach(([className, confidence]) => {
            if (confidence > maxConfidence) {
                maxConfidence = confidence;
                topClassification = className;
            }
        });
        
        // Display only the top classification
        if (topClassification) {
            const classificationDiv = document.createElement('div');
            classificationDiv.className = 'flex items-center justify-between p-4 bg-gray-800/50 rounded-lg border border-sky-500/30';
            
            const textColor = 'text-sky-300';
            const bgColor = 'bg-sky-500';
            
            classificationDiv.innerHTML = `
                <span class="${textColor} font-semibold text-lg">${topClassification}</span>
                <div class="flex items-center gap-4">
                    <div class="w-32 bg-gray-700 rounded-full h-3">
                        <div class="${bgColor} h-3 rounded-full transition-all duration-500" style="width: ${maxConfidence}%"></div>
                    </div>
                    <span class="${textColor} text-lg font-mono font-bold">${maxConfidence}%</span>
                </div>
            `;
            
            allClassifications.appendChild(classificationDiv);
        }
    }
    
    // Display patient information
    if (patientInfo) {
        patientInfo.innerHTML = '';
        
        const patientFields = [
            { label: t('patientId'), value: results.PatientID },
            { label: t('patientName'), value: results.PatientName },
            { label: t('sex'), value: results.PatientSex },
            { label: t('age'), value: results.Age },
            { label: t('weight'), value: results.PatientWeight }
        ];
        
        patientFields.forEach(field => {
            const fieldDiv = document.createElement('div');
            fieldDiv.className = 'flex justify-between';
            fieldDiv.innerHTML = `
                <span class="text-sky-400">${field.label}</span>
                <span class="text-sky-100">${field.value || 'Unknown'}</span>
            `;
            patientInfo.appendChild(fieldDiv);
        });
    }
}

// PDF Report Generation Function
function generatePDFReport() {
    console.log('🔄 Starting professional PDF generation...');
    
    // Simple validation
    if (!analysisResults) {
        alert('Please run an analysis first to generate a report.');
        return;
    }

    try {
        // Simple jsPDF access that works
        const doc = new window.jspdf.jsPDF();
        
        // Header
        doc.setFontSize(20);
        doc.setTextColor(41, 128, 185);
        doc.text('MEDICAL IMAGE ANALYSIS REPORT', 20, 20);
        
        // Divider line
        doc.setDrawColor(41, 128, 185);
        doc.setLineWidth(0.5);
        doc.line(20, 25, 190, 25);
        
        // Report Information
        doc.setFontSize(12);
        doc.setTextColor(0, 0, 0);
        doc.text('Report Generated:', 20, 35);
        doc.text(new Date().toLocaleString(), 70, 35);
        
        doc.text('File Name:', 20, 42);
        doc.text(analysisResults.fileName || 'Unknown', 70, 42);
        
        doc.text('File Size:', 20, 49);
        doc.text(analysisResults.fileSize || 'Unknown', 70, 49);
        
        // Patient Information Section
        doc.setFontSize(16);
        doc.setTextColor(41, 128, 185);
        doc.text('PATIENT INFORMATION', 20, 65);
        
        doc.setFontSize(12);
        doc.setTextColor(0, 0, 0);
        let yPos = 75;
        
        const patientData = [
            ['Patient ID:', analysisResults.PatientID || 'Unknown'],
            ['Patient Name:', analysisResults.PatientName || 'Unknown'],
            ['Sex:', analysisResults.PatientSex || 'Unknown'],
            ['Age:', analysisResults.Age || 'Unknown'],
            ['Weight:', analysisResults.PatientWeight || 'Unknown']
        ];
        
        patientData.forEach(([label, value]) => {
            doc.text(label, 20, yPos);
            doc.text(value, 70, yPos);
            yPos += 7;
        });
        
        // Primary Diagnosis Section
        yPos += 10;
        doc.setFontSize(16);
        doc.setTextColor(41, 128, 185);
        doc.text('PRIMARY DIAGNOSIS', 20, yPos);
        
        yPos += 10;
        doc.setFontSize(14);
        
        // Color code diagnosis
        const diagnosis = analysisResults.Diagnosis;
        if (diagnosis === 'Normal') {
            doc.setTextColor(34, 139, 34);
        } else if (diagnosis === 'TB Positive' || diagnosis === 'Lung Cancer') {
            doc.setTextColor(220, 20, 60);
        } else {
            doc.setTextColor(255, 140, 0);
        }
        
        doc.text(`${diagnosis}`, 20, yPos);
        doc.setTextColor(0, 0, 0);
        doc.setFontSize(12);
        doc.text(`Confidence: ${analysisResults.Confidence}%`, 20, yPos + 7);
        
        // Top Classification Section
        yPos += 25;
        doc.setFontSize(16);
        doc.setTextColor(41, 128, 185);
        doc.text('TOP CLASSIFICATION', 20, yPos);
        
        yPos += 10;
        doc.setFontSize(12);
        doc.setTextColor(0, 0, 0);
        
        if (analysisResults.AllClassConfidences) {
            // Find the highest confidence classification
            let maxConfidence = 0;
            let topClassification = '';
            
            Object.entries(analysisResults.AllClassConfidences).forEach(([className, confidence]) => {
                if (confidence > maxConfidence) {
                    maxConfidence = confidence;
                    topClassification = className;
                }
            });
            
            // Display only the top classification
            if (topClassification) {
                // Class name
                doc.text(`${topClassification}:`, 25, yPos);
                
                // Percentage value (right-aligned)
                doc.text(`${maxConfidence}%`, 170, yPos);
                
                // Draw progress bar with better positioning
                const barStartX = 90;
                const barWidth = 70;
                const barHeight = 4;
                const fillWidth = (maxConfidence / 100) * barWidth;
                
                // Background bar (light gray)
                doc.setFillColor(220, 220, 220);
                doc.rect(barStartX, yPos - 2.5, barWidth, barHeight, 'F');
                
                // Progress bar border
                doc.setDrawColor(180, 180, 180);
                doc.setLineWidth(0.2);
                doc.rect(barStartX, yPos - 2.5, barWidth, barHeight);
                
                // Fill bar with primary color
                doc.setFillColor(41, 128, 185); // Blue for top classification
                
                if (fillWidth > 0) {
                    doc.rect(barStartX, yPos - 2.5, fillWidth, barHeight, 'F');
                }
                
                yPos += 15; // Space after the classification
            }
        }
        
        // Clinical Notes Section
        yPos += 10;
        doc.setFontSize(16);
        doc.setTextColor(41, 128, 185);
        doc.text('CLINICAL NOTES', 20, yPos);
        
        yPos += 10;
        doc.setFontSize(10);
        doc.setTextColor(100, 100, 100);
        
        const clinicalNotes = [
            'This analysis was performed using an AI-powered medical image classification system.',
            'The system uses advanced deep learning algorithms trained on medical imaging data.',
            '',
            'IMPORTANT DISCLAIMERS:',
            '• This AI system is designed to assist healthcare professionals',
            '• Results should not be used as the sole basis for medical diagnosis',
            '• Always consult qualified healthcare providers for proper medical interpretation',
            '• Clinical correlation and additional testing may be required',
            '• This report should be reviewed by a qualified radiologist or physician'
        ];
        
        clinicalNotes.forEach(note => {
            if (note === '') {
                yPos += 5;
            } else {
                doc.text(note, 20, yPos);
                yPos += 5;
            }
        });
        
        // Footer
        doc.setFontSize(8);
        doc.setTextColor(150, 150, 150);
        doc.text('Generated by Medical Image AI Classification System', 20, 280);
        doc.text(`Report ID: ${Date.now()}`, 20, 285);
        
        // Save the PDF
        const fileName = `Medical_Analysis_Report_${analysisResults.PatientID || 'Unknown'}_${new Date().toISOString().split('T')[0]}.pdf`;
        doc.save(fileName);
        
        // Show success message
        setTimeout(() => {
            alert('Professional medical report downloaded successfully!');
        }, 500);
        
    } catch (error) {
        console.error('PDF Error:', error);
        alert('PDF generation failed. Error: ' + error.message);
    }
}

// Main Analysis Function - Optimized to prevent double upload
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select a file first');
        return;
    }
    
    if (isAnalyzing) {
        console.log('Analysis already in progress, ignoring click');
        return;
    }
    
    try {
        isAnalyzing = true;
        
        // Disable the analyze button to prevent multiple clicks
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');
        
        // Show progress page
        showPage('page4');
        
        // Single API call for complete analysis
        updateProgress(10, 'Preparing analysis...');
        const results = await analyzeImageAPI(selectedFile);
        
        // Display results after short delay for better UX
        setTimeout(() => {
            displayResults(results);
            showPage('page5');
            isAnalyzing = false;
        }, 1000);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'An error occurred during analysis');
        showPage('page3'); // Go back to upload page
        
        // Reset progress and flags
        updateProgress(0, 'Ready to analyze...');
        isAnalyzing = false;
        
        // Re-enable analyze button if we have a selected file
        if (selectedFile) {
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }
}

// New Analysis Function
function startNewAnalysis() {
    console.log('🔄 Starting new analysis...');
    
    // Reset all states
    resetFileSelection();
    updateProgress(0, 'Ready to analyze...');
    
    // Go back to upload page
    showPage('page3');
    console.log('✅ Navigated back to upload page');
}

// Initialize the application and attach all event listeners
document.addEventListener('DOMContentLoaded', () => {
    console.log('Medical Image Classification System initialized');
    
    // ====================================
    // SELECT ALL DOM ELEMENTS FIRST
    // ====================================
    
    // Page elements
    pages = {
        page1: document.getElementById('page1'),
        page2: document.getElementById('page2'),
        page3: document.getElementById('page3'),
        page4: document.getElementById('page4'),
        page5: document.getElementById('page5')
    };
    
    // Language Selection Elements
    languageSelect = document.getElementById('languageSelect');
    nextBtn = document.getElementById('nextBtn');
    
    // File Upload Elements  
    fileDropArea = document.getElementById('file-drop-area');
    fileInput = document.getElementById('file-input');
    fileInfo = document.getElementById('file-info');
    fileName = document.getElementById('file-name');
    fileSize = document.getElementById('file-size');
    removeFileBtn = document.getElementById('remove-file');
    errorMessage = document.getElementById('error-message');
    analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtnText = document.getElementById('analyze-btn-text');
    
    // Progress Elements
    progressBar = document.getElementById('progress-bar');
    progressText = document.getElementById('progress-text');
    
    // Results Elements
    resultImage = document.getElementById('result-image');
    resultFilename = document.getElementById('result-filename');
    primaryDiagnosis = document.getElementById('primary-diagnosis');
    confidenceScore = document.getElementById('confidence-score');
    allClassifications = document.getElementById('all-classifications');
    patientInfo = document.getElementById('patient-info');
    downloadReportBtn = document.getElementById('download-report');
    newAnalysisBtn = document.getElementById('new-analysis');
    
    console.log('DOM Elements selected:', {
        downloadReportBtn: !!downloadReportBtn,
        newAnalysisBtn: !!newAnalysisBtn,
        analyzeBtn: !!analyzeBtn
    });
    
    // Debug button properties
    if (downloadReportBtn) {
        console.log('Download button found:', {
            element: downloadReportBtn,
            classes: downloadReportBtn.className,
            style: downloadReportBtn.style.cssText,
            visible: downloadReportBtn.offsetParent !== null
        });
    }
    
    if (newAnalysisBtn) {
        console.log('New Analysis button found:', {
            element: newAnalysisBtn,
            classes: newAnalysisBtn.className,
            style: newAnalysisBtn.style.cssText,
            visible: newAnalysisBtn.offsetParent !== null
        });
    }
    
    // ====================================
    // INITIALIZE APPLICATION STATE
    // ====================================
    
    // Ensure we start on page 1
    showPage('page1');
    
    // Reset all form states
    if (languageSelect) {
        languageSelect.value = '';
    }
    if (nextBtn) {
        nextBtn.disabled = true;
    }
    resetFileSelection();
    
    // Initialize with English language
    updateLanguage();
    
    // ====================================
    // ATTACH ALL EVENT LISTENERS HERE
    // ====================================
    
    // Page Navigation Event Listeners
    document.getElementById('next-to-page2')?.addEventListener('click', () => {
        console.log('START ANALYSIS button clicked');
        showPage('page2');
    });

    document.getElementById('back-to-page1')?.addEventListener('click', () => {
        console.log('Back to page1 clicked');
        showPage('page1');
    });

    document.getElementById('back-to-page2')?.addEventListener('click', () => {
        console.log('Back to page2 clicked');
        showPage('page2');
    });

    // Language Selection Logic
    languageSelect?.addEventListener('change', function() {
        currentLanguage = this.value;
        console.log(`🌐 Language changed to: ${currentLanguage}`);
        
        if (this.value) {
            nextBtn.disabled = false;
            nextBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            
            // Update all text on the page immediately
            updateLanguage();
        } else {
            nextBtn.disabled = true;
            nextBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
    });

    nextBtn?.addEventListener('click', function() {
        if (!this.disabled) {
            // Ensure language is updated before proceeding
            updateLanguage();
            showPage('page3');
        }
    });

    // File Upload Event Listeners
    fileInput?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelection(file);
        }
    });

    // Note: Removed click handler for fileDropArea - the label element handles this naturally
    // This prevents the double-click issue while maintaining functionality

    fileDropArea?.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileDropArea.classList.add('dragover');
    });

    fileDropArea?.addEventListener('dragleave', (e) => {
        e.preventDefault();
        fileDropArea.classList.remove('dragover');
    });

    fileDropArea?.addEventListener('drop', (e) => {
        e.preventDefault();
        fileDropArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    removeFileBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFileSelection();
    });

    // Analysis Event Listener
    analyzeBtn?.addEventListener('click', analyzeImage);

    // Action Button Event Listeners
    if (newAnalysisBtn) {
        newAnalysisBtn.addEventListener('click', () => {
            console.log('New Analysis button clicked');
            startNewAnalysis();
        });
        console.log('✅ New Analysis button listener attached');
    } else {
        console.error('❌ New Analysis button not found');
    }
    
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', function() {
            console.log('Download Report button clicked - SIMPLE VERSION');
            generatePDFReport();
        });
        console.log('✅ Download Report button listener attached - SIMPLE VERSION');
    } else {
        console.error('❌ Download Report button not found');
    }
    
    console.log('🎯 Application initialization complete');
    
    // Add a global test function to check buttons
    window.testButtons = function() {
        console.log('🧪 Testing buttons manually...');
        
        const downloadBtn = document.getElementById('download-report');
        const newBtn = document.getElementById('new-analysis');
        
        console.log('Download button:', {
            found: !!downloadBtn,
            visible: downloadBtn?.offsetParent !== null,
            clickable: downloadBtn?.style.pointerEvents !== 'none'
        });
        
        console.log('New Analysis button:', {
            found: !!newBtn,
            visible: newBtn?.offsetParent !== null,
            clickable: newBtn?.style.pointerEvents !== 'none'
        });
        
        if (downloadBtn) {
            console.log('Manually triggering download...');
            generatePDFReport();
        }
        
        if (newBtn) {
            console.log('Manually triggering new analysis...');
            startNewAnalysis();
        }
    };
    
    console.log('💡 Use window.testButtons() in console to test buttons manually');
});

// Handle browser back/forward buttons
window.addEventListener('popstate', (e) => {
    // Handle browser navigation if needed
    console.log('Navigation state changed');
});