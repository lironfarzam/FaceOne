importScripts('tf.min.js', 'tf-converter.min.js');

let faceNetModel = null;

async function initTensorFlow() {
    try {
        await tf.ready();
        await tf.setBackend('webgl');
        const backend = await tf.backend();
        
        if (backend && backend.setWebGLFlag) {
            backend.setWebGLFlag('WEBGL_FORCE_F16_TEXTURES', true);
            backend.setWebGLFlag('WEBGL_VERSION', 2);
        }
        
        postMessage({
            type: 'TF_INITIALIZED',
            success: true,
            info: {
                version: tf.version.tfjs,
                backend: tf.getBackend()
            }
        });
    } catch (error) {
        postMessage({
            type: 'TF_INITIALIZED',
            success: false,
            error: error.message
        });
    }
}

async function loadModel(modelPath) {
    try {
        faceNetModel = await tf.loadGraphModel(modelPath);
        
        // Warm up
        const dummyInput = tf.zeros([1, 160, 160, 3]);
        const warmupResult = await faceNetModel.predict(dummyInput);
        warmupResult.dispose();
        dummyInput.dispose();
        
        postMessage({
            type: 'MODEL_LOADED',
            success: true,
            modelInfo: {
                inputShape: faceNetModel.inputs[0].shape,
                outputShape: faceNetModel.outputs[0].shape
            }
        });
    } catch (error) {
        postMessage({
            type: 'MODEL_LOADED',
            success: false,
            error: error.message
        });
    }
}

async function generateEmbedding(imageData) {
    if (!faceNetModel) {
        postMessage({
            type: 'EMBEDDING_GENERATED',
            success: false,
            error: 'Model not loaded'
        });
        return;
    }

    try {
        const tensor = tf.tidy(() => {
            const img = tf.tensor(imageData, [160, 160, 4]);
            const rgb = img.slice([0, 0, 0], [-1, -1, 3]);
            return rgb.expandDims(0).toFloat().div(127.5).sub(1);
        });

        const embedding = await faceNetModel.predict(tensor);
        const embeddingData = await embedding.data();
        
        // L2 normalize
        const normalizedEmbedding = tf.tidy(() => {
            const embeddingTensor = tf.tensor1d(embeddingData);
            return tf.div(embeddingTensor, tf.norm(embeddingTensor));
        });
        
        const finalEmbedding = await normalizedEmbedding.data();
        
        // Cleanup
        tensor.dispose();
        embedding.dispose();
        normalizedEmbedding.dispose();
        
        postMessage({
            type: 'EMBEDDING_GENERATED',
            success: true,
            embedding: Array.from(finalEmbedding)
        });
    } catch (error) {
        postMessage({
            type: 'EMBEDDING_GENERATED',
            success: false,
            error: error.message
        });
    }
}

self.onmessage = async function(e) {
    switch (e.data.type) {
        case 'INIT':
            await initTensorFlow();
            break;
        case 'LOAD_MODEL':
            await loadModel(e.data.modelPath);
            break;
        case 'GENERATE_EMBEDDING':
            await generateEmbedding(e.data.imageData);
            break;
    }
}; 