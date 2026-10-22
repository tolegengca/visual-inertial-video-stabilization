#if canImport(CoreMotion)
import CoreMotion
import Foundation

public final class MotionStabilizer: @unchecked Sendable {
    private let manager = CMMotionManager()
    private let queue = OperationQueue()
    private var samples: [(TimeInterval, CMQuaternion)] = []
    private let lock = NSLock()

    public init() {
        queue.name = "StabilizationPipeline.CoreMotion"
        queue.maxConcurrentOperationCount = 1
    }

    public func start() {
        guard manager.isDeviceMotionAvailable, !manager.isDeviceMotionActive else { return }
        manager.deviceMotionUpdateInterval = 1.0 / 120.0
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] motion, _ in
            guard let self, let motion else { return }
            self.lock.lock()
            self.samples.append((motion.timestamp, motion.attitude.quaternion))
            if self.samples.count > 480 { self.samples.removeFirst(self.samples.count - 480) }
            self.lock.unlock()
        }
    }

    public func stop() { manager.stopDeviceMotionUpdates() }

    public func rollDelta(from start: TimeInterval, to end: TimeInterval) -> Float? {
        lock.lock(); defer { lock.unlock() }
        guard let a = samples.min(by: { abs($0.0-start) < abs($1.0-start) }),
              let b = samples.min(by: { abs($0.0-end) < abs($1.0-end) }) else { return nil }
        let rollA = atan2(2 * (a.1.w*a.1.z + a.1.x*a.1.y), 1 - 2 * (a.1.y*a.1.y + a.1.z*a.1.z))
        let rollB = atan2(2 * (b.1.w*b.1.z + b.1.x*b.1.y), 1 - 2 * (b.1.y*b.1.y + b.1.z*b.1.z))
        return Float(rollB-rollA)
    }
}
#endif
