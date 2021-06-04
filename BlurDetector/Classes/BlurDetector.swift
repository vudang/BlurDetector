
//
//  BlurDetector.swift
//  BlurDetector
//
//  Created by Petr Bobák on 03/10/2019.
//  Copyright © 2019 Oneprove. All rights reserved.
//

import CoreML
import Vision
@_exported import MLPatchExtractor

public struct BlurDetectorResult {
    /// Evaluated image (single patch).
    public var image: UIImage
    
    /// Predicted class label.
    public var classLabel: String
    
    /// Confidence of predicted class label.
    public var confidence: Double
    
    /// Rectange that specifies the size and position of image (patch).
    public var rect: CGRect
}

extension MobileNetV2_BlurDetectorInput: MLPatchExtractorProtocol  {
    static func create(image: CVPixelBuffer) -> MLFeatureProvider {
        return MobileNetV2_BlurDetectorInput(input_1: image)
    }
}

public class BlurDetector {
    private static let patchSize = CGSize(width: 224, height: 224)
    private static let coreMLModel = MobileNetV2_BlurDetector()
    
    public static let defaultMaskFactor: CGFloat = 0.7
    public static let defaultNumberOfPatches: Int = 50
    
    /**
     Predicts blurriness probability of `image`.
     
     - Parameters:
        - image: The source image to be evaluated.
        - patches: The number of patches to generate (in case of `uniform` the number of generated patches can be slightly larger or smaller in favor of uniform coverage).
        - sampling: The method to sample patches (`random` or `uniform`, for more details see `MLPatchSampling`).
        - maskRectangle: The image area to generate patches from.
        - completion: The completion block called after prediction is completed. Returns aggregated blurriness probability and results for each extracted patch.
     */
    public static func evaluate(image: UIImage, patches: Int = defaultNumberOfPatches, sampling: MLPatchSampling = .random, maskRectangle: CGRect, completion: @escaping (Double, [BlurDetectorResult]) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            var start = CACurrentMediaTime()

            guard let (patches, rects) =
                MLPatchExtractor<MobileNetV2_BlurDetectorInput>.extract(image: image, patchSize: self.patchSize, count: patches, sampling: sampling,
                maskRectangle: maskRectangle) as? ([MobileNetV2_BlurDetectorInput], [CGRect]) else {
                print("No patches")
                return
            }
            
            var end = CACurrentMediaTime()
            let patchExtractionTime = end - start
            
//            // Single patch
//            guard let pixelBuffer = patches[0].pixelBuffer() else {
//                return
//            }
//            start = CACurrentMediaTime()
//            let prediction = try? model.prediction(image: pixelBuffer)
//            end = CACurrentMediaTime()
//            print("Single prediction took: \(end - start) seconds")
            
            // Batch of patches
            start = CACurrentMediaTime()
            let predictions = try? self.coreMLModel.predictions(inputs: patches)
            end = CACurrentMediaTime()
            let batchPredictionTime = end - start

            var aggeregatedLabelCount = [
                "blurred" : 0,
                "focused" : 0
            ]
            
            var perPatchResults = [BlurDetectorResult]()
            
            for (i, prediction) in (predictions ?? []).enumerated() {
                let patch = patches[i]
                let rect = rects[i]
                
                // Aggregated total confidence of each class (blurred, focused) from all patches
                guard let currentValue = aggeregatedLabelCount[prediction.classLabel] else {
                    print("Unknown label")
                    return
                }
                aggeregatedLabelCount[prediction.classLabel] = currentValue + 1
                
                // Create return array
                perPatchResults.append(BlurDetectorResult(image: UIImage(pixelBuffer: patch.input_1)!,
                                                          classLabel: prediction.classLabel,
                                                          confidence: prediction.Identity[prediction.classLabel] ?? -1,
                                                          rect: rect))
            }
            
            let probability = Double(aggeregatedLabelCount["blurred"]!) / Double(aggeregatedLabelCount["blurred"]! + aggeregatedLabelCount["focused"]!)
            
            print("\(aggeregatedLabelCount.description)" +
                "\nPath extraction: " + String(format:"%.3f", patchExtractionTime) + "s" +
                "\nBatch prediction: " + String(format:"%.3f", batchPredictionTime) + "s")
            
            completion(probability, perPatchResults)
        }
    }
    
    /**
     Predicts blurriness probability of `image`.
     
     - Parameters:
        - image: The source image to be evaluated.
        - patches: The number of patches to generate (in case of `uniform` the number of generated patches can be slightly larger or smaller in favor of uniform coverage).
        - sampling: The method to sample patches (`random` or `uniform`, for more details see `MLPatchSampling`).
        - maskFactor: The percent of the image area to generate from (centered to center o the input image).
        - completion: The completion block called after prediction is completed. Returns aggregated blurriness probability and results for each extracted patch.
     */
    public static func evaluate(image: UIImage, patches: Int = defaultNumberOfPatches, sampling: MLPatchSampling = .random, maskFactor: CGFloat = defaultMaskFactor, completion: @escaping (Double, [BlurDetectorResult]) -> Void) {
        
        var maskRectangle = CGRect(origin: .zero, size: image.size)
        if maskFactor < 1.0 {
           maskRectangle = CGRect(x: image.size.width / 2 - (maskFactor * image.size.width) / 2,
                                  y: image.size.height / 2 - (maskFactor * image.size.height) / 2,
                                  width: maskFactor * image.size.width,
                                  height: maskFactor * image.size.height)
        }
        
        evaluate(image: image, patches: patches, sampling: sampling, maskRectangle: maskRectangle, completion: completion)
    }
}
