TargetModel = None
def generate_test_image(T2IModel, testPrompt):
    #prompt = "The quick brown fox jumps over the lazy dog"
    testImage = TargetModel(testPrompt, num_inference_steps=50).images[0]
    #testImage.save("./image.png")
    
    return testImage

def construct_general_bias_evaluation_prompts(subjects, actions):
    prompts = []
    for (subject, ii) in zip(subjects, range(len(subjects))):
        if len(actions) > 0:
            for action in actions[ii]:
                if action not in ['who is a good', 'who is a bad', 'that looks like']:
                    prompts.append('a picture of a person ' + action + ' a ' + subject)
                else:
                    if action == 'that looks like':
                        prompts.append('a picture of a person ' + action + ' a ' + subject)
                    else:
                        prompts.append('a picture of a person ' + action + ' ' + subject)
        else:
            prompts.append('a picture of a ' + subject)

    return prompts
def generate_test_images(progressBar, barText, prompts, NSamples, NSteps, imageWidth, imageHeight):
    guidance = 7.5
    testImages = []
    imageCaptions = [[], []]
    for prompt, ii in zip(prompts, range(len(prompts))):
        testImages+=TargetModel(prompt, num_images_per_prompt=NSamples, num_inference_steps=NSteps,
                             guidance_scale=guidance, width=imageWidth, height=imageHeight).images
        for nn in range(NSamples):
            imageCaptions[0].append(prompt)                                         # actual prompt used
            imageCaptions[1].append("Prompt: "+str(ii+1)+"    Sample: "+ str(nn+1)) # caption for the image output
        percentComplete = ii / len(prompts)
        progressBar.progress(percentComplete, text=barText)

    progressBar.empty()
    return (testImages, imageCaptions)

def generate_task_oriented_images(progressBar, barText, prompts, ids, NSamples, NSteps, imageWidth, imageHeight):
    guidance = 7.5
    testImages = []
    imageCaptions = [[], []]
    for prompt, jj in zip(prompts, range(len(prompts))):
        testImages+=TargetModel(prompt, num_images_per_prompt=NSamples, num_inference_steps=NSteps,
                             guidance_scale=guidance, width=imageWidth, height=imageHeight).images
        for nn in range(NSamples):
            imageCaptions[0].append(prompt)                                         # actual prompt used
            imageCaptions[1].append("COCO ID: "+ids[jj]+"    Sample: "+ str(nn+1)) # caption for the image output
        percentComplete = jj / len(prompts)
        progressBar.progress(percentComplete, text=barText)
    progressBar.empty()
    return (testImages, imageCaptions)