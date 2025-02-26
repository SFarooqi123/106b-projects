// Generated by gencpp from file stdr_msgs/RfidSensorMsg.msg
// DO NOT EDIT!


#ifndef STDR_MSGS_MESSAGE_RFIDSENSORMSG_H
#define STDR_MSGS_MESSAGE_RFIDSENSORMSG_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Pose2D.h>

namespace stdr_msgs
{
template <class ContainerAllocator>
struct RfidSensorMsg_
{
  typedef RfidSensorMsg_<ContainerAllocator> Type;

  RfidSensorMsg_()
    : maxRange(0.0)
    , angleSpan(0.0)
    , signalCutoff(0.0)
    , frequency(0.0)
    , frame_id()
    , pose()  {
    }
  RfidSensorMsg_(const ContainerAllocator& _alloc)
    : maxRange(0.0)
    , angleSpan(0.0)
    , signalCutoff(0.0)
    , frequency(0.0)
    , frame_id(_alloc)
    , pose(_alloc)  {
  (void)_alloc;
    }



   typedef float _maxRange_type;
  _maxRange_type maxRange;

   typedef float _angleSpan_type;
  _angleSpan_type angleSpan;

   typedef float _signalCutoff_type;
  _signalCutoff_type signalCutoff;

   typedef float _frequency_type;
  _frequency_type frequency;

   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _frame_id_type;
  _frame_id_type frame_id;

   typedef  ::geometry_msgs::Pose2D_<ContainerAllocator>  _pose_type;
  _pose_type pose;





  typedef boost::shared_ptr< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> const> ConstPtr;

}; // struct RfidSensorMsg_

typedef ::stdr_msgs::RfidSensorMsg_<std::allocator<void> > RfidSensorMsg;

typedef boost::shared_ptr< ::stdr_msgs::RfidSensorMsg > RfidSensorMsgPtr;
typedef boost::shared_ptr< ::stdr_msgs::RfidSensorMsg const> RfidSensorMsgConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::stdr_msgs::RfidSensorMsg_<ContainerAllocator1> & lhs, const ::stdr_msgs::RfidSensorMsg_<ContainerAllocator2> & rhs)
{
  return lhs.maxRange == rhs.maxRange &&
    lhs.angleSpan == rhs.angleSpan &&
    lhs.signalCutoff == rhs.signalCutoff &&
    lhs.frequency == rhs.frequency &&
    lhs.frame_id == rhs.frame_id &&
    lhs.pose == rhs.pose;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::stdr_msgs::RfidSensorMsg_<ContainerAllocator1> & lhs, const ::stdr_msgs::RfidSensorMsg_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace stdr_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "a210bd881966e92d87b9dcab3e305128";
  }

  static const char* value(const ::stdr_msgs::RfidSensorMsg_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xa210bd881966e92dULL;
  static const uint64_t static_value2 = 0x87b9dcab3e305128ULL;
};

template<class ContainerAllocator>
struct DataType< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "stdr_msgs/RfidSensorMsg";
  }

  static const char* value(const ::stdr_msgs::RfidSensorMsg_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Rfid sensor description\n"
"float32 maxRange\n"
"float32 angleSpan\n"
"float32 signalCutoff\n"
"\n"
"float32 frequency\n"
"\n"
"string frame_id\n"
"geometry_msgs/Pose2D pose # sensor pose, relative to robot center\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Pose2D\n"
"# Deprecated\n"
"# Please use the full 3D pose.\n"
"\n"
"# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.\n"
"\n"
"# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.\n"
"\n"
"\n"
"# This expresses a position and orientation on a 2D manifold.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 theta\n"
;
  }

  static const char* value(const ::stdr_msgs::RfidSensorMsg_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.maxRange);
      stream.next(m.angleSpan);
      stream.next(m.signalCutoff);
      stream.next(m.frequency);
      stream.next(m.frame_id);
      stream.next(m.pose);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct RfidSensorMsg_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::stdr_msgs::RfidSensorMsg_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::stdr_msgs::RfidSensorMsg_<ContainerAllocator>& v)
  {
    s << indent << "maxRange: ";
    Printer<float>::stream(s, indent + "  ", v.maxRange);
    s << indent << "angleSpan: ";
    Printer<float>::stream(s, indent + "  ", v.angleSpan);
    s << indent << "signalCutoff: ";
    Printer<float>::stream(s, indent + "  ", v.signalCutoff);
    s << indent << "frequency: ";
    Printer<float>::stream(s, indent + "  ", v.frequency);
    s << indent << "frame_id: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.frame_id);
    s << indent << "pose: ";
    s << std::endl;
    Printer< ::geometry_msgs::Pose2D_<ContainerAllocator> >::stream(s, indent + "  ", v.pose);
  }
};

} // namespace message_operations
} // namespace ros

#endif // STDR_MSGS_MESSAGE_RFIDSENSORMSG_H
